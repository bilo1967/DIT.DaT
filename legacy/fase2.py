#!/usr/bin/env python3
"""
Fase 2 - Versione con validazione avanzata
Merge consecutivi + filtro durata con validazione input e operazioni speaker
"""

import argparse
import json
import os
import sys
import time
import yaml
from collections import defaultdict

# Funzioni comuni
import common_utils


def validate_input_data(data):
    """Valida la struttura del JSON di input"""
    if not isinstance(data, dict):
        raise ValueError("✗ Errore: input non è un file JSON valido")
    
    if 'segments' not in data:
        raise ValueError("✗ Errore: nel file JSON di input manca della chiave 'segments'")
    
    if not isinstance(data['segments'], list):
        raise ValueError("✗ Errore: il campo 'segments' deve essere una lista")
    
    required_fields = ['speaker', 'start', 'end']
    for i, seg in enumerate(data['segments']):
        if not isinstance(seg, dict):
            raise ValueError(f"✗ Errore: il segmento {i} non è un oggetto valido")
        
        for field in required_fields:
            if field not in seg:
                raise ValueError(f"✗ Errore: nel segmento {i} manca il campo '{field}'")
        
        # Valida tipi e valori
        if not isinstance(seg['start'], (int, float)) or seg['start'] < 0:
            raise ValueError(f"✗ Errore: il segmento {i} ha un 'start' non valido: {seg['start']}")
        
        if not isinstance(seg['end'], (int, float)) or seg['end'] < 0:
            raise ValueError(f"✗ Errore: il segmento {i} ha un 'end' non valido: {seg['end']}")
        
        if seg['end'] <= seg['start']:
            raise ValueError(f"✗ Errore: il segmento {i} ha end ({seg['end']}) <= start ({seg['start']})")
    
    return True

def parse_speaker_operations(merge_str, rename_str, drop_str):
    """Parsa le operazioni sugli speaker con validazione"""
    merge_speakers = {}
    rename_speakers = {}
    drop_speakers = []
    
    try:
        # Parse merge
        if merge_str and merge_str.strip():
            for pair in merge_str.split(','):
                pair = pair.strip()
                if not pair:
                    continue
                if '=' not in pair:
                    raise ValueError(f"✗ Errore: formato merge non valido: '{pair}'. Usa 'SPEAKER_A=SPEAKER_B'")
                
                sources, target = pair.split('=', 1)
                sources = sources.strip()
                target = target.strip()
                
                if not sources:
                    raise ValueError(f"✗ Errore: nomi speaker sorgente vuoti in: '{pair}'")
                if not target:
                    raise ValueError(f"✗ Errore: nome speaker target vuoto in: '{pair}'")
                
                for source in sources.split('+'):
                    source = source.strip()
                    if not source:
                        continue
                    merge_speakers[source] = target
        
        # Parse rename
        if rename_str and rename_str.strip():
            for pair in rename_str.split(','):
                pair = pair.strip()
                if not pair:
                    continue
                if '=' not in pair:
                    raise ValueError(f"✗ Errore: formato rename non valido: '{pair}'. Usa 'SPEAKER_A=NuovoNome'")
                
                old_name, new_name = pair.split('=', 1)
                old_name = old_name.strip()
                new_name = new_name.strip()
                
                if not old_name:
                    raise ValueError(f"✗ Errore: nome speaker vecchio vuoto in: '{pair}'")
                if not new_name:
                    raise ValueError(f"✗ Errore: nome speaker nuovo vuoto in: '{pair}'")
                
                rename_speakers[old_name] = new_name
        
        # Parse drop
        if drop_str and drop_str.strip():
            for speaker in drop_str.split(','):
                speaker = speaker.strip()
                if speaker:
                    drop_speakers.append(speaker)
            
    except Exception as e:
        raise ValueError(f"✗ Errore nel parsing operazioni sugli speaker: {e}")
    
    return merge_speakers, rename_speakers, drop_speakers

def validate_speaker_operations(merge_speakers, rename_speakers, drop_speakers, available_speakers):
    """Valida che le operazioni sugli speaker siano consistenti"""
    warnings = []
    errors = []
    
    all_operation_speakers = set()
    all_operation_speakers.update(merge_speakers.keys())
    all_operation_speakers.update(merge_speakers.values())
    all_operation_speakers.update(rename_speakers.keys())
    all_operation_speakers.update(rename_speakers.values())
    all_operation_speakers.update(drop_speakers)
    
    # Controlla speaker inesistenti
    non_existent = all_operation_speakers - set(available_speakers)
    if non_existent:
        warnings.append(f"Speaker nelle operazioni che non esistono nei dati: {sorted(non_existent)}")
    
    # Controlla conflitti
    for speaker in merge_speakers:
        if speaker in rename_speakers:
            errors.append(f"Speaker '{speaker}' presente sia in merge che in rename")
        if speaker in drop_speakers:
            errors.append(f"Speaker '{speaker}' presente sia in merge che in drop")
    
    # Controlla cicli in merge
    for source, target in merge_speakers.items():
        current = target
        visited = {source}
        while current in merge_speakers:
            if current in visited:
                errors.append(f"Ciclo di merge rilevato: {source} -> ... -> {current}")
                break
            visited.add(current)
            current = merge_speakers[current]
    
    # Controlla che non si rinomini uno speaker in un altro già esistente (se non è un merge)
    for old_name, new_name in rename_speakers.items():
        if new_name in available_speakers and new_name not in merge_speakers.values():
            warnings.append(f"Rename '{old_name}' -> '{new_name}': il nuovo nome è già uno speaker esistente")
    
    return errors, warnings

def get_output_json_path(project_dir, args):
    """Ottiene il percorso del file JSON di output"""
    if args.output_json:
        return args.output_json
    
    # Percorso predefinito nella project directory
    return os.path.join(project_dir, "fase2_filtered.json")


def get_parameters_from_config(project_dir, args):
    """Ottiene i parametri dal config.yaml se non specificati nella CLI"""
    config = common_utils.load_config(project_dir)
    if not config:
        return args
    
    phase2_config = config.get('phase2', {})
    
    # Parametri con fallback al config
    if args.min_pause is None and 'min_pause' in phase2_config:
        args.min_pause = phase2_config['min_pause']
        print(f"Pausa minima tra segmenti dello stesso speaker: {args.min_pause}")
    
    if args.min_duration is None and 'min_duration' in phase2_config:
        args.min_duration = phase2_config['min_duration']
        print(f"Durata minima di un segmento isolato: {args.min_duration}")
    
    if not args.merge_speakers and 'merge_speakers' in phase2_config:
        args.merge_speakers = phase2_config['merge_speakers']
        print(f"Speaker uniti: {args.merge_speakers}")
    
    if not args.rename_speakers and 'rename_speakers' in phase2_config:
        args.rename_speakers = phase2_config['rename_speakers']
        print(f"Speaker rinominati: {args.rename_speakers}")
    
    if not args.drop_speakers and 'drop_speakers' in phase2_config:
        args.drop_speakers = phase2_config['drop_speakers']
        print(f"Speaker scartati: {args.drop_speakers}")
    
    return args

def group_merges(merge_rules):
    groups = defaultdict(set)
    for src, target in merge_rules.items():
        groups[target].add(src)
    return groups

def process_segments(input_json, params):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Inizializza struttura e statistiche
    speakers = defaultdict(lambda: {
        "display_name": None,
        "segments": [],
        "original_segments": []
    })
    
    stats = {
        "total_segments_pre_merge": 0,
        "total_segments_post_merge": 0,
        "total_segments_post_filter": 0,
        "segments_merged": 0,
        "short_segments_removed": 0,
        "avg_duration_all_speakers": 0,
        "speakers": defaultdict(lambda: {
            "segments_pre_merge": 0,
            "segments_post_merge": 0,
            "segments_merged": 0,
            "short_segments_removed": 0,
            "segments_final": 0,
            "avg_duration_pre_merge": 0,
            "avg_duration_post_merge": 0
        })
    }

    metadata = data.get('metadata', {
        'audio_file': input_json.replace('.json', '.wav'),
        'checksum': 'N/A'
    })

    # 1. Preprocessamento speaker
    for seg in data['segments']:
        if 'speaker' not in seg:
            continue

        speaker = seg['speaker']
        
        # Applica regole speaker
        if speaker in params['merge_speakers']:
            speaker = params['merge_speakers'][speaker]
        if speaker in params['rename_speakers']:
            speakers[speaker]["display_name"] = params['rename_speakers'][speaker]
        if speaker in params['drop_speakers']:
            continue

        # Aggiungi segmento con flag iniziale
        seg_id = seg.get('id', f"{speaker}_{len(speakers[speaker]['segments'])+1}")
        new_seg = {
            "id": seg_id,
            "speaker": speaker,
            "start": seg['start'],
            "end": seg['end'],
            "duration": seg['end'] - seg['start'],
            "original_ids": seg.get('original_ids', [seg_id]),
            "processing_flag": "unmodified"
        }
        
        speakers[speaker]['segments'].append(new_seg)
        speakers[speaker]['original_segments'].append(new_seg.copy())
        stats['speakers'][speaker]['segments_pre_merge'] += 1
        stats['total_segments_pre_merge'] += 1

    # 2. Merge segmenti consecutivi (stesso speaker)
    for speaker in speakers:
        segments = speakers[speaker]['segments']
        if not segments:
            continue

        # Ordina per tempo di inizio
        segments.sort(key=lambda x: x['start'])
        original_count = len(segments)

        # Fase di merge
        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            
            # Calcola gap tra segmenti (negativo se sovrapposti)
            gap = seg['start'] - last['end']
            
            if gap < params['min_pause']:  # Include sovrapposizioni (gap < 0)
                # Unisci segmenti
                last['end'] = max(last['end'], seg['end'])  # Considera sovrapposizioni
                last['duration'] = last['end'] - last['start']
                last['original_ids'].extend(seg['original_ids'])
                last['processing_flag'] = 'aggregated'
            else:
                # Segmento separato
                merged.append(seg)
        
        speakers[speaker]['segments'] = merged
        merged_count = len(merged)

        # Calcolo statistiche merge
        durations_pre = [s['duration'] for s in speakers[speaker]['original_segments']]
        durations_post = [s['duration'] for s in merged]
        
        stats['speakers'][speaker].update({
            "segments_post_merge": merged_count,
            "segments_merged": original_count - merged_count,
            "avg_duration_pre_merge": sum(durations_pre) / len(durations_pre) if durations_pre else 0,
            "avg_duration_post_merge": sum(durations_post) / len(durations_post) if durations_post else 0
        })
        stats['total_segments_post_merge'] += merged_count
        stats['segments_merged'] += (original_count - merged_count)

    # 3. Filtro segmenti troppo brevi
    segments_removed = 0

    for speaker in speakers:
        segments = speakers[speaker]['segments']
        if not segments:
            continue

        initial_count = len(segments)
        
        # Filtra segmenti per durata
        filtered_segments = []
        for seg in segments:
            if seg['duration'] >= params['min_duration']:
                filtered_segments.append(seg)
            else:
                seg['processing_flag'] = 'removed'
                segments_removed += 1
                stats['speakers'][speaker]['short_segments_removed'] += 1
                if params.get('verbose'):
                    print(f"Rimosso segmento {seg['id']} (durata: {seg['duration']:.3f}s < {params['min_duration']}s)")
        
        stats['speakers'][speaker]['segments_final'] = stats['speakers'][speaker]['segments_post_merge'] - stats['speakers'][speaker]['short_segments_removed']
        speakers[speaker]['segments'] = filtered_segments

    # Aggiorna statistiche globali
    stats['short_segments_removed'] = segments_removed
    stats['total_segments_post_filter'] = stats['total_segments_post_merge'] - segments_removed
    stats['total_segments_final'] = stats['total_segments_post_filter']

    # 4. Calcolo durata media globale (solo segmenti finali)
    all_durations = [
        s['duration'] 
        for speaker in speakers.values() 
        for s in speaker['segments']
    ]
    stats['avg_duration_all_speakers'] = (
        sum(all_durations) / len(all_durations) 
        if all_durations else 0
    )

    # 5. Genera indice cross-speaker
    segment_index = {}
    for speaker, data in speakers.items():
        for idx, seg in enumerate(data['segments']):
            segment_index[seg['id']] = {
                "speaker": speaker,
                "segment_idx": idx
            }

    return {
        "metadata": {
            **metadata,
            "parameters": { 
                **{k: v for k, v in params.items() if k != "merge_speakers"},
                "merge_speakers_groups": [
                    {"from": sorted(list(sources)), "into": target}
                    for target, sources in group_merges(params["merge_speakers"]).items()
                ]
            },
            "stats": stats
        },
        "speakers": dict(speakers),
        "segment_index": segment_index
    }


def update_config_with_fase2_paths(project_dir, output_json_path, params):
    """Aggiorna config.yaml con i percorsi e parametri di fase2"""
    config = common_utils.load_config(project_dir)
    if not config:
        return False

    try:
        if 'paths' not in config:
            config['paths'] = {}

        config['paths']['fase2_filtered'] = os.path.basename(output_json_path)

        # Salva parametri usati per riproducibilità
        if 'phase2' not in config:
            config['phase2'] = {}

        config['phase2']['last_parameters'] = {
            'min_pause': params.get('min_pause'),
            'min_duration': params.get('min_duration'),
            'merge_speakers': params.get('merge_speakers'),
            'rename_speakers': params.get('rename_speakers'),
            'drop_speakers': params.get('drop_speakers'),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        config_path = os.path.join(project_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        return True
    except Exception as e:
        print(f"AVVISO: Impossibile aggiornare config.yaml: {e}")
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 2 - Merge segmenti consecutivi e filtro durata",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--project-dir", required=True, help="Directory del progetto")
    parser.add_argument("--input-json", help="Input JSON dalla Fase 1 (opzionale, default: project-dir/fase1_unified.json)")
    parser.add_argument("--output-json", help="Output JSON (opzionale, default: project-dir/fase2_filtered.json)")
    parser.add_argument("--min-pause", type=float, 
                       help="Soglia per unire segmenti consecutivi (secondi).\n"
                            "Segmenti dello stesso speaker separati da una pausa INFERIORE a questo valore\n"
                            "verranno uniti in un unico turno di parola.\n"
                            "Include automaticamente le sovrapposizioni (gap negativo).\n"
                            "[Default: dal config.yaml o 1.5]")
    parser.add_argument("--min-duration", type=float,
                       help="Durata minima per mantenere un segmento (secondi).\n"
                            "Dopo il merge, i segmenti con durata INFERIORE a questo valore\n"
                            "verranno rimossi.\n"
                            "[Default: dal config.yaml o 0.5]")
    parser.add_argument("--merge-speakers", type=str, default="",
                       help="Unisci speaker (es. 'SPEAKER_01=SPEAKER_02,SPEAKER_03=Ciccio')")
    parser.add_argument("--rename-speakers", type=str, default="",
                       help="Rinomina speaker (es. 'SPEAKER_01=Intervistatore,SPEAKER_02=Ospite')")
    parser.add_argument("--drop-speakers", type=str, default="",
                       help="Speaker da rimuovere (es. 'SPEAKER_00,SPEAKER_04')")
    parser.add_argument("--verbose", action="store_true", help="Mostra log dettagliato")
    parser.add_argument("--force", action="store_true", help="Sovrascrive file esistenti senza chiedere conferma")

    args = parser.parse_args()

    # Verifica directory progetto
    if not os.path.exists(args.project_dir):
        print(f"✗ Errore: directory di progetto non trovata: {args.project_dir}")
        sys.exit(1)

    # Ottieni parametri dal config se non specificati
    args = get_parameters_from_config(args.project_dir, args)

    # Ottieni percorso input
    input_json_path = common_utils.get_input_json_path(args.project_dir, args, "fase1_unified.json")
    if not input_json_path or not os.path.exists(input_json_path):
        print(f"✗ Errore: file JSON di input non trovato: {input_json_path}")
        print("Assicurati di avere già eseguito fase1.py")
        sys.exit(1)

    # Validazione input JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        validate_input_data(input_data)
        print("✓ Validazione struttura input JSON completata")
        
    except Exception as e:
        print(f"✗ Errore: input JSON non valido: {e}")
        sys.exit(1)

    # Parsing e validazione operazioni speaker
    try:
        merge_speakers, rename_speakers, drop_speakers = parse_speaker_operations(
            args.merge_speakers, args.rename_speakers, args.drop_speakers
        )
        print("✓ Parsing operazioni sugli speaker completato")
        
    except ValueError as e:
        print(f"✗ Errore nel parsing delle operazioni sugli speaker: {e}")
        sys.exit(1)

    # Validazione consistenza operazioni speaker
    try:
        available_speakers = set(seg['speaker'] for seg in input_data['segments'])
        errors, warnings = validate_speaker_operations(
            merge_speakers, rename_speakers, drop_speakers, available_speakers
        )
        
        if warnings:
            print("! Avvisi nelle operazioni sugli speaker:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if errors:
            print("✗ Errori nelle operazioni sugli speaker:")
            for error in errors:
                print(f"  - {error}")
            print("Impossibile procedere - correggere gli errori")
            sys.exit(1)
            
        print("✓ Validazione operazioni speaker completata")
        
    except Exception as e:
        print(f"✗ Errore nella validazione operazioni speaker: {e}")
        sys.exit(1)

    # Ottieni percorso output
    output_json_path = get_output_json_path(args.project_dir, args)

    # Verifica sovrascrittura
    if not common_utils.check_existing_output(output_json_path, args.force):
        sys.exit(1)

    start_time = time.time()

    # Valori di default se non specificati né in CLI né in config
    min_pause = args.min_pause if args.min_pause is not None else 1.5
    min_duration = args.min_duration if args.min_duration is not None else 0.5

    params = {
        "min_pause": min_pause,
        "min_duration": min_duration,
        "merge_speakers": merge_speakers,
        "rename_speakers": rename_speakers,
        "drop_speakers": drop_speakers,
        "verbose": args.verbose
    }

    print(f"Elaborazione: {input_json_path}")
    print(f"Parametri: min_pause={min_pause}s, min_duration={min_duration}s")
    
    try:
        result = process_segments(input_json_path, params)
    except Exception as e:
        print(f"✗ Errore durante l'elaborazione: {e}")
        sys.exit(1)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


    config_updated = update_config_with_fase2_paths(args.project_dir, output_json_path, params)
    if config_updated:
        print("✓ Configurazione aggiornata")
    else:
        print("! Configurazione non aggiornata")




    
    # Report finale
    stats = result['metadata']['stats']
    print(f"\n=== FASE 2 COMPLETATA ===")
    print(f"Input:  {input_json_path}")
    print(f"Output: {output_json_path}")

    print(f"\nRapporto segmenti:")
    print(f"    iniziali       {stats['total_segments_pre_merge']:5d}")
    print(f"    uniti         -{stats['segments_merged']:5d}")
    print(f"    dopo il merge ={stats['total_segments_post_merge']:5d}")
    print(f"    brevi rimossi -{stats['short_segments_removed']:5d}")
    print(f"    finali        ={stats['total_segments_final']:5d}")
    
    # Conteggio processing flags
    processing_stats = defaultdict(int)
    for speaker_data in result['speakers'].values():
        for seg in speaker_data['segments']:
            processing_stats[seg.get('processing_flag', 'unmodified')] += 1
    
    print(f"di cui:")
    for flag, count in processing_stats.items():
        flag = "aggregati" if flag == "aggregated" else "originali" 
        print(f"    {flag:14s} {count:5d}")
    
    print(f"\nDurata media:      {stats['avg_duration_all_speakers']:5.2f}s\n")


    # Tempo totale
    elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)
