#!/usr/bin/env python3
"""
FASE 1 - Unifica blocchi fase0 e applica mappatura speaker
Versione con validazione avanzata
"""

import argparse
import json
import os
import sys
import time
import re
import yaml
import yaml
from collections import defaultdict
from pathlib import Path

# Funzioni comuni
import common_utils


def get_speaker_map_path(project_dir, args):
    """Ottiene il percorso della mappa speaker (argomento o config)"""
    if args.speaker_map:
        return args.speaker_map
    
    # Cerca nel config
    config = common_utils.load_config(project_dir)
    if config and 'paths' in config and 'speaker_map' in config['paths']:
        map_path = os.path.join(project_dir, config['paths']['speaker_map'])
        if os.path.exists(map_path):
            return map_path
    
    # Fallback al percorso predefinito
    default_path = os.path.join(project_dir, "speakers_map.txt")
    if os.path.exists(default_path):
        return default_path
    
    return None

def parse_speaker_map(map_path):
    """Parsa il file di mappatura speaker con regex robusta e supporto commenti inline"""
    speaker_map = {}
    line_number = 0
    parsing_errors = []

    with open(map_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            line = line.strip()

            # Rimuovi commenti (tutto dopo #)
            line = re.sub(r'#.*$', '', line).strip()
            if not line:
                continue

            # Regex robusta: permette spazi, trattini, underscore nei nomi
            # Formato: BLOCK_XX.SPEAKER_YY => NOME_GLOBALE
            match = re.match(r'^(BLOCK_\d+\.\w+)\s*=>\s*([\w\-_]+)\s*$', line)
            if match:
                block_speaker = match.group(1)
                global_speaker = match.group(2)
                speaker_map[block_speaker] = global_speaker
            else:
                parsing_errors.append(f"Riga {line_number}: formato non valido - '{line}'")

    return speaker_map, parsing_errors


def normalize_speaker_id(speaker_id):
    """Normalizza BLOCK_X.SPEAKER_YY in BLOCK_XX.SPEAKER_YY mantenendo il numero originale"""
    match = re.match(r'BLOCK_(\d+)\.(SPEAKER_\d+)', speaker_id)
    if match:
        block_num = int(match.group(1))  # Converti a numero (rimuove zeri iniziali)
        speaker_name = match.group(2)
        return f"BLOCK_{block_num:02d}.{speaker_name}"  # Standardizza a 2 cifre
    return speaker_id

def validate_speaker_map(speaker_map, blocks_data):
    """
    Valida la mappa speaker contro i dati effettivi dei blocchi
    Restituisce: (is_valid, errors, warnings, analysis)
    """
    errors = []
    warnings = []
    analysis = []  # Nuovo: per informazioni non problematiche

    # Normalizza la mappa speaker
    normalized_speaker_map = {normalize_speaker_id(k): v for k, v in speaker_map.items()}

    # Raccolta di tutti gli speaker effettivamente presenti nei blocchi
    actual_speakers = set()
    for block_id, block_data in blocks_data:
        block_speakers = set(seg['speaker'] for seg in block_data['segments'])
        for speaker in block_speakers:
            actual_speakers.add(f"BLOCK_{block_id:02d}.{speaker}")

    # Raccolta di tutti gli speaker nella mappa (normalizzati)
    mapped_speakers = set(normalized_speaker_map.keys())

    # Controllo 1: Speaker nella mappa che non esistono nei dati → WARNING
    non_existent_speakers = mapped_speakers - actual_speakers
    if non_existent_speakers:
        warnings.append(f"Speaker nella mappa che non esistono nei blocchi: {sorted(non_existent_speakers)}")

    # Controllo 2: Speaker nei blocchi che non sono nella mappa → ERROR
    unmapped_speakers = actual_speakers - mapped_speakers
    if unmapped_speakers:
        errors.append(f"Speaker nei blocchi non mappati: {sorted(unmapped_speakers)}")

    # Controllo 3: Nomi globali duplicati → ANALYSIS (non warning!)
    global_speaker_counts = defaultdict(list)
    for block_speaker, global_speaker in normalized_speaker_map.items():
        global_speaker_counts[global_speaker].append(block_speaker)

    duplicate_globals = {global_speaker: speakers
                        for global_speaker, speakers in global_speaker_counts.items()
                        if len(speakers) > 1}

    if duplicate_globals:
        analysis.append("Accorpamenti speaker rilevati:")
        for global_speaker, block_speakers in duplicate_globals.items():
            analysis.append(f"  {global_speaker} <- {block_speakers}")

    return len(errors) == 0, errors, warnings, analysis, normalized_speaker_map


def load_block_data(project_dir):
    """Carica tutti i dati dei blocchi fase0 con ordinamento robusto"""
    blocks_data = []
    metadata = None
    
    # Carica metadata globale
    metadata_path = os.path.join(project_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"ERRORE: Impossibile leggere metadata.json: {e}")
            metadata = None
    
    # Directory blocks
    blocks_dir = os.path.join(project_dir, "blocks")
    if not os.path.exists(blocks_dir):
        print(f"ERRORE: Directory blocks non trovata: {blocks_dir}")
        return [], None
    
    # Pattern robusto per identificare directory dei blocchi
    block_dirs = []
    for item in Path(blocks_dir).iterdir():
        if item.is_dir() and item.name.startswith("BLOCK_"):
            try:
                # Estrai numero blocco in modo robusto
                block_num = int(item.name.split('_')[1])
                block_dirs.append((block_num, item))
            except (ValueError, IndexError):
                print(f"ATTENZIONE: Ignorata directory con nome non valido: {item.name}")
                continue
    
    if not block_dirs:
        print(f"ERRORE: Nessuna directory BLOCK_* valida trovata in {blocks_dir}")
        return [], None
    
    # Ordina per numero blocco
    block_dirs.sort(key=lambda x: x[0])
    
    # Verifica sequenza continua
    expected_sequence = list(range(len(block_dirs)))
    actual_sequence = [block_num for block_num, _ in block_dirs]
    
    if actual_sequence != expected_sequence:
        print(f"ATTENZIONE: Sequenza blocchi non continua: {actual_sequence}")
        print("I blocchi saranno processati nell'ordine numerico trovato")
    
    # Carica dati di ogni blocco
    for block_num, block_dir in block_dirs:
        # Prova diversi pattern di nomi file
        json_patterns = [
            block_dir / f"block_{block_num:02d}.json",
            block_dir / f"block_{block_num}.json",
            block_dir / "block.json"
        ]
        
        json_path = None
        for pattern in json_patterns:
            if pattern.exists():
                json_path = pattern
                break
        
        if not json_path:
            print(f"ERRORE: File JSON non trovato per blocco {block_num} in {block_dir}")
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                block_data = json.load(f)
                blocks_data.append((block_num, block_data))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"ERRORE: Impossibile leggere {json_path}: {e}")
            continue
    
    return blocks_data, metadata


def unify_blocks(blocks_data, speaker_map):
    """Unifica i blocchi applicando la mappatura speaker normalizzata"""
    all_segments = []

    # Normalizza la mappa
    normalized_speaker_map = {normalize_speaker_id(k): v for k, v in speaker_map.items()}

    for block_id, block_data in blocks_data:
        segments = block_data.get('segments', [])

        for seg in segments:
            original_speaker = seg['speaker']
            block_speaker_id = f"BLOCK_{block_id:02d}.{original_speaker}"

            # Applica mappatura speaker normalizzata
            if block_speaker_id in normalized_speaker_map:
                mapped_speaker = normalized_speaker_map[block_speaker_id]
            else:
                # Se non mappato, mantieni originale
                mapped_speaker = original_speaker

            # Crea nuovo segmento
            new_segment = seg.copy()
            new_segment["speaker"] = mapped_speaker
            new_segment["original_speaker"] = original_speaker

            all_segments.append(new_segment)

    # Ordina per ID
    all_segments.sort(key=lambda x: x["id"])
    return all_segments


def calculate_segment_type_counts(segments):
    """Calcola conteggi tipi di segmento"""
    type_counts = defaultdict(int)
    for seg in segments:
        type_counts[seg['type']] += 1
    return dict(type_counts)



def main():

    parser = argparse.ArgumentParser(description="FASE 1 - Unifica blocchi fase0")
    parser.add_argument("--project-dir", required=True, help="Directory del progetto (output di fase0)")
    parser.add_argument("--speaker-map", help="File speakers_map.txt (opzionale, default: project-dir/speakers_map.txt)")
    parser.add_argument("--output-json", help="File JSON di output (opzionale, default: project-dir/fase1_unified.json)")
    parser.add_argument("--force", action="store_true", help="Sovrascrive file esistenti senza chiedere conferma")
    
    args = parser.parse_args()
    
    # Verifica directory progetto
    if not os.path.exists(args.project_dir):
        print(f"ERRORE: Directory di progetto non trovata: {args.project_dir}")
        return 1
    
    # Ottieni percorso mappa speaker
    map_path = get_speaker_map_path(args.project_dir, args)
    if not map_path or not os.path.exists(map_path):
        print(f"ERRORE: File mappa speaker non trovato: {map_path}")
        print("Assicurati di aver compilato speakers_map.txt dopo fase0")
        return 1
    
    print(f"Caricamento mappatura speaker: {map_path}")
    speaker_map, parsing_errors = parse_speaker_map(map_path)
    
    if parsing_errors:
        print("ERRORI DI PARSING nella mappa speaker:")
        for error in parsing_errors:
            print(f"  - {error}")
        
        user_input = input("Procedere comunque? [y/N] ")
        if user_input.lower() != 'y':
            print("Operazione annullata - correggere la mappa speaker")
            return 1
    
    print(f"Mappature caricate: {len(speaker_map)}")
    
    print("Caricamento dati blocchi...")
    blocks_data, metadata = load_block_data(args.project_dir)
    print(f"Blocchi trovati: {len(blocks_data)}")
    
    if not blocks_data:
        print("ERRORE: Nessun blocco trovato nella directory di progetto")
        return 1
    
    # Validazione mappa speaker
    print("Validazione mappa speaker...")
    is_valid, errors, warnings, analysis, normalized_speaker_map = validate_speaker_map(speaker_map, blocks_data)

    # Mostra analysis (informazioni normali)
    if analysis:
        print("Analisi mappatura:")
        for item in analysis:
            print(f"  {item}")
        print()

    # Mostra warnings (problemi non critici)
    if warnings:
        print("AVVISI DI VALIDAZIONE:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    # Mostra errors (problemi critici)
    if errors:
        print("ERRORI DI VALIDAZIONE CRITICI:")
        for error in errors:
            print(f"  - {error}")

        user_input = input("\nProcedere comunque? [y/N] ")
        if user_input.lower() != 'y':
            print("Operazione annullata - correggere la mappa speaker")
            return 1
        else:
            print("Proseguimento forzato nonostante gli errori...")


    # Determina percorso output
    if args.output_json:
        output_path = args.output_json
    else:
        output_path = os.path.join(args.project_dir, "fase1_unified.json")

    # Verifica sovrascrittura
    if not common_utils.check_existing_output(output_path, args.force):
        return 1

    start_time = time.time()

    print("Unificazione blocchi...")
    unified_segments = unify_blocks(blocks_data, normalized_speaker_map)
    print(f"Segmenti totali: {len(unified_segments)}")
    
    # Verifica che gli ID siano univoci
    all_ids = [seg['id'] for seg in unified_segments]
    unique_ids = set(all_ids)
    if len(all_ids) != len(unique_ids):
        print(f"ATTENZIONE: Ci sono ID duplicati! Totali: {len(all_ids)}, Unici: {len(unique_ids)}")
        # Troviamo i duplicati
        from collections import Counter
        id_counts = Counter(all_ids)
        duplicates = [id for id, count in id_counts.items() if count > 1]
        print(f"ID duplicati: {duplicates[:10]}")  # Mostra solo i primi 10
    
    # Calcola statistiche speaker
    speakers = set(seg['speaker'] for seg in unified_segments)
    print(f"Speaker univoci finali: {len(speakers)}")
    
    
    # Prepara metadata finale
    if metadata:
        final_metadata = {
            "source_file": metadata.get("source_file", ""),
            "wav_file": metadata.get("wav_file", ""),
            "total_duration": metadata.get("total_duration", 0),
            "block_duration": metadata.get("block_duration", 0),
            "num_blocks": metadata.get("num_blocks", 0),
            "processing_time_sec": metadata.get("processing_time", 0),
            "num_speakers": len(speakers),
            "total_segments": len(unified_segments),
            "avg_confidence": 0.5
        }
    else:
        final_metadata = {
            "source_file": "",
            "wav_file": "",
            "total_duration": 0,
            "block_duration": 0,
            "num_blocks": 0,
            "processing_time_sec": 0,
            "num_speakers": len(speakers),
            "total_segments": len(unified_segments),
            "avg_confidence": 0.5
        }
    
    # Calcola confidenza media
    confidences = [seg['confidence'] for seg in unified_segments if 'confidence' in seg]
    if confidences:
        final_metadata['avg_confidence'] = round(sum(confidences) / len(confidences), 4)
    
    # Struttura finale
    output_data = {
        "metadata": final_metadata,
        "segments": unified_segments,
        "segment_type_counts": calculate_segment_type_counts(unified_segments)
    }
    
    # Salva JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== FASE 1 COMPLETATA ===")
    print(f"File generato: {output_path}")
    print(f"Segmenti: {len(unified_segments)}")
    print(f"Speaker: {len(speakers)}")
    print(f"Blocchi processati: {len(blocks_data)}")
    
    if len(blocks_data) == 1:
        print("Modalita single-block: mappatura nomi applicata")
    else:
        print("Modalita multi-block: unificazione completata")
    
    print(f"Speaker finali: {', '.join(sorted(speakers))}")
    
    # Mostra statistiche mappatura
    mapped_count = sum(1 for seg in unified_segments if seg['speaker'] != seg.get('original_speaker', seg['speaker']))
    if mapped_count > 0:
        print(f"Segmenti mappati: {mapped_count}/{len(unified_segments)}")
    
    # Tempo totale
    elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)
    return 0


if __name__ == "__main__":
    main()
