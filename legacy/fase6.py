#!/usr/bin/env python3
"""
FASE 6 - Validazione e rigenerazione sottotitoli dopo revisione
Valida il file SRT revisionato e rigenera tutti gli output finali
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Funzioni comuni
import common_utils

def parse_srt_file(srt_path):
    """
    Parsa file SRT restituendo lista di sottotitoli
    Formato: [{number, start, end, start_sec, end_sec, text, raw_text}, ...]
    """
    subtitles = []
    
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"File SRT non trovato: {srt_path}")
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Dividi in blocchi di sottotitoli
    blocks = content.split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            number = int(lines[0].strip())
            time_line = lines[1].strip()
            
            # Parse timestamps SRT (HH:MM:SS,mmm --> HH:MM:SS,mmm)
            time_match = re.match(r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)', time_line)
            if not time_match:
                raise ValueError(f"Formato timestamp non valido: {time_line}")
            
            start_str = time_match.group(1)
            end_str = time_match.group(2)
            
            # Converti in secondi
            start_sec = srt_timestamp_to_seconds(start_str)
            end_sec = srt_timestamp_to_seconds(end_str)
            
            # Testo (tutte le righe rimanenti)
            text_lines = lines[2:]
            raw_text = '\n'.join(text_lines)
            text = raw_text.strip()
            
            subtitles.append({
                'number': number,
                'start': start_str,
                'end': end_str,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'text': text,
                'raw_text': raw_text
            })
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Errore parsing blocco SRT: {lines[0] if lines else 'N/A'} - {e}")
    
    return subtitles

def srt_timestamp_to_seconds(timestamp):
    """Converte timestamp SRT (HH:MM:SS,mmm) in secondi"""
    try:
        # Sostituisci virgola con punto per i millisecondi
        timestamp = timestamp.replace(',', '.')
        parts = timestamp.split(':')
        
        if len(parts) != 3:
            raise ValueError("Formato timestamp non valido")
            
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError) as e:
        raise ValueError(f"Timestamp non valido: {timestamp} - {e}")

def extract_speaker_from_text(text):
    """
    Estrae tag speaker dal testo usando regex
    Restituisce (speaker, text_without_tag, error)
    """
    # Regex per [SPEAKER] all'inizio del testo
    match = re.match(r'^\s*\[([^\]]+)\]\s*(.*)$', text, re.DOTALL)
    if match:
        speaker = match.group(1).strip()
        clean_text = match.group(2).strip()

        # Controlla se il tag speaker contiene caratteri non validi (virgole, punti e virgola, etc.)
        if re.search(r'[,;]', speaker):
            return None, text, f"Il tag speaker contiene caratteri non validi: '{speaker}'"

        return speaker, clean_text, None
    else:
        return None, text, "Manca il tag speaker (formato: [SPEAKER])"

def validate_subtitles(subtitles, expected_speakers):
    """
    Validazione completa dei sottotitoli
    Restituisce: (is_valid, errors, warnings, validated_subs)
    """
    errors = []
    warnings = []
    validated_subs = []

    # 1. Controllo tag speaker e struttura
    speakers_found = set()

    for i, sub in enumerate(subtitles):
        sub_errors = []
        sub_warnings = []

        # Controllo tag speaker
        speaker, clean_text, speaker_error = extract_speaker_from_text(sub['text'])

        if speaker_error:
            sub_errors.append(speaker_error)
        elif speaker:
            sub['speaker'] = speaker
            sub['clean_text'] = clean_text
            speakers_found.add(speaker)

            # Controllo speaker sconosciuto
            if expected_speakers and speaker not in expected_speakers:
                sub_warnings.append(f"Speaker sconosciuto: '{speaker}'")

        # Controllo durata valida
        if sub['end_sec'] <= sub['start_sec']:
            sub_errors.append(f"Timestamp fine <= inizio ({sub['start']} --> {sub['end']})")

        # Controllo testo vuoto
        if not clean_text.strip():
            sub_warnings.append("Testo vuoto dopo tag speaker")

        # Aggiungi a validated_subs solo se ha un speaker valido
        if speaker and not speaker_error:
            validated_subs.append(sub)

        # Raccolta errori/warning con contesto
        if sub_errors:
            errors.append({
                'subtitle_number': sub['number'],
                'start_time': sub['start'],
                'errors': sub_errors,
                'text_preview': sub['text'][:100] + ('...' if len(sub['text']) > 100 else '')
            })

        if sub_warnings:
            warnings.append({
                'subtitle_number': sub['number'],
                'start_time': sub['start'],
                'warnings': sub_warnings,
                'text_preview': sub['text'][:100] + ('...' if len(sub['text']) > 100 else '')
            })

    # 2. Controllo speaker mancanti
    if expected_speakers:
        missing_speakers = expected_speakers - speakers_found
        if missing_speakers:
            warnings.append({
                'type': 'global',
                'message': f"Speaker attesi senza sottotitoli: {sorted(missing_speakers)}"
            })

    # 3. Controllo sovrapposizioni per speaker
    overlap_errors = check_speaker_overlaps(validated_subs)
    errors.extend(overlap_errors)

    # 4. Controllo finale: errori bloccanti
    has_blocking_errors = any(len(item['errors']) > 0 for item in errors if 'errors' in item)

    return (not has_blocking_errors), errors, warnings, validated_subs


def check_speaker_overlaps(subtitles):
    """Controlla sovrapposizioni temporali per ogni speaker"""
    errors = []
    
    # Raggruppa per speaker
    speaker_segments = defaultdict(list)
    for sub in subtitles:
        speaker_segments[sub['speaker']].append(sub)
    
    for speaker, segments in speaker_segments.items():
        # Ordina per tempo di inizio
        segments.sort(key=lambda x: x['start_sec'])
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            # Controlla sovrapposizione
            if current['end_sec'] > next_seg['start_sec']:
                errors.append({
                    'type': 'overlap',
                    'speaker': speaker,
                    'subtitle_1': current['number'],
                    'subtitle_2': next_seg['number'],
                    'time_1': f"{current['start']} --> {current['end']}",
                    'time_2': f"{next_seg['start']} --> {next_seg['end']}",
                    'overlap_sec': round(current['end_sec'] - next_seg['start_sec'], 2)
                })
    
    return errors

def get_expected_speakers(project_dir):
    """Ottiene la lista degli speaker attesi da fase2_filtered.json"""
    fase2_path = os.path.join(project_dir, "fase2_filtered.json")
    if not os.path.exists(fase2_path):
        print(f"AVVISO: File {fase2_path} non trovato - validazione speaker disabilitata")
        return None
    
    try:
        with open(fase2_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return set(data['speakers'].keys())
    except Exception as e:
        print(f"AVVISO: Impossibile caricare speaker attesi: {e}")
        return None

def generate_output_files(validated_subs, output_dir):
    """Genera tutti i file di output finali"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Raggruppa per speaker
    speaker_subs = defaultdict(list)
    for sub in validated_subs:
        speaker_subs[sub['speaker']].append(sub)
    
    # Genera file per ogni speaker
    for speaker, subs in speaker_subs.items():
        generate_speaker_files(speaker, subs, output_dir)
    
    # Genera file combinati
    generate_combined_files(validated_subs, output_dir)
    
    return len(speaker_subs)

def generate_speaker_files(speaker, subs, output_dir):
    """Genera file per un singolo speaker"""
    speaker_safe = speaker.replace(' ', '_')
    
    # SRT per speaker
    srt_path = os.path.join(output_dir, f"{speaker_safe}.srt")
    generate_srt_file(subs, srt_path, include_speaker_tag=False)
    
    # TXT con timestamp
    txt_timestamp_path = os.path.join(output_dir, f"{speaker_safe}_with_timestamps.txt")
    generate_txt_file(subs, txt_timestamp_path, with_timestamps=True, include_speaker_tag=False)
    
    # TXT pulito
    txt_clean_path = os.path.join(output_dir, f"{speaker_safe}_clean.txt")
    generate_txt_file(subs, txt_clean_path, with_timestamps=False, include_speaker_tag=False)

def generate_combined_files(subs, output_dir):
    """Genera file combinati con tutti gli speaker"""
    # Ordina tutti i sottotitoli per tempo
    subs.sort(key=lambda x: x['start_sec'])
    
    # SRT combinato
    combined_srt_path = os.path.join(output_dir, "podcast_complete_validated.srt")
    generate_srt_file(subs, combined_srt_path, include_speaker_tag=True)
    
    # TXT combinato con timestamp
    combined_txt_path = os.path.join(output_dir, "podcast_with_timestamps_validated.txt")
    generate_txt_file(subs, combined_txt_path, with_timestamps=True, include_speaker_tag=True)
    
    # TXT combinato pulito
    combined_clean_path = os.path.join(output_dir, "podcast_clean_validated.txt")
    generate_txt_file(subs, combined_clean_path, with_timestamps=False, include_speaker_tag=True)

def generate_srt_file(subs, output_path, include_speaker_tag=True):
    """Genera file SRT"""
    srt_content = []
    
    for i, sub in enumerate(subs, 1):
        # Costruisci testo
        if include_speaker_tag:
            text = f"[{sub['speaker']}] {sub['clean_text']}"
        else:
            text = sub['clean_text']
        
        srt_content.append(f"{i}")
        srt_content.append(f"{sub['start']} --> {sub['end']}")
        srt_content.append(text)
        srt_content.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_content))

def generate_txt_file(subs, output_path, with_timestamps=True, include_speaker_tag=True):
    """Genera file TXT"""
    txt_content = []
    
    for sub in subs:
        # Costruisci linea
        parts = []
        
        if with_timestamps:
            parts.append(format_timestamp_readable(sub['start_sec']))
        
        if include_speaker_tag:
            parts.append(f"[{sub['speaker']}]")
        
        parts.append(sub['clean_text'])
        
        txt_content.append(" ".join(parts))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_content))

def format_timestamp_readable(seconds):
    """Formatta timestamp in [HH:MM:SS] per TXT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:05.2f}]"

def save_validation_report(errors, warnings, output_dir):
    """Salva report di validazione"""

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "validation_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== REPORT VALIDAZIONE SOTTOTITOLI ===\n\n")
        
        # Errori
        if errors:
            f.write("ERRORI TROVATI:\n")
            f.write("-" * 50 + "\n")
            for error in errors:
                if 'type' in error and error['type'] == 'global':
                    f.write(f"GLOBALE: {error['message']}\n\n")
                elif 'type' in error and error['type'] == 'overlap':
                    f.write(f"SOVRAPPOSIZIONE: Speaker '{error['speaker']}'\n")
                    f.write(f"  Sottotitolo {error['subtitle_1']}: {error['time_1']}\n")
                    f.write(f"  Sottotitolo {error['subtitle_2']}: {error['time_2']}\n")
                    f.write(f"  Sovrapposizione: {error['overlap_sec']} secondi\n\n")
                else:
                    f.write(f"Sottotitolo {error['subtitle_number']} ({error['start_time']}):\n")
                    for err in error['errors']:
                        f.write(f"  - {err}\n")
                    f.write(f"  Anteprima: {error['text_preview']}\n\n")
        else:
            f.write("NESSUN ERRORE TROVATO ✓\n\n")
        
        # Warning
        if warnings:
            f.write("AVVISI:\n")
            f.write("-" * 50 + "\n")
            for warning in warnings:
                if 'type' in warning and warning['type'] == 'global':
                    f.write(f"GLOBALE: {warning['message']}\n\n")
                else:
                    f.write(f"Sottotitolo {warning['subtitle_number']} ({warning['start_time']}):\n")
                    for warn in warning['warnings']:
                        f.write(f"  - {warn}\n")
                    f.write(f"  Anteprima: {warning['text_preview']}\n\n")
        else:
            f.write("NESSUN AVVISO TROVATO ✓\n\n")
    
    return report_path


def get_speaker_stats(validated_subs):
    """Calcola statistiche per speaker"""
    speaker_stats = defaultdict(int)
    for sub in validated_subs:
        speaker_stats[sub['speaker']] += 1
    return dict(speaker_stats)


def main():
    parser = argparse.ArgumentParser(
        description="FASE 6 - Validazione e rigenerazione sottotitoli dopo revisione\n"
                   "Valida il file SRT revisionato e rigenera output finali",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--project-dir", required=True, 
           help="Directory del progetto")
    parser.add_argument("--input", 
           help="File SRT revisionato (default: project-dir/subs/podcast_complete.srt)")
    parser.add_argument("--output-dir", 
           help="Directory output (default: project-dir/subs_final)")
    parser.add_argument("--force", action="store_true", 
           help="Sovrascrive output esistenti senza chiedere conferma")
    parser.add_argument("--quiet", action="store_true",
           help="Non mostra il report di validazione a schermo")
    
    args = parser.parse_args()
    
    # Verifica directory progetto
    if not os.path.exists(args.project_dir):
        print(f"ERRORE: Directory di progetto non trovata: {args.project_dir}")
        return 1
    
    # Determina percorso input SRT
    if args.input:
        srt_path = args.input
    else:
        srt_path = os.path.join(args.project_dir, "subs", "podcast_complete.srt")
    
    # Determina directory output
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.project_dir, "subs_final")
    
    # Verifica sovrascrittura
    if not common_utils.check_existing_output(output_dir, args.force):
        return 1
    
    start_time = time.time()
    
    print("=== FASE 6 - VALIDAZIONE SOTTOTITOLI ===")
    print(f"Input SRT: {srt_path}")
    print(f"Output directory: {output_dir}")
    print(f"Directory progetto: {args.project_dir}")
    
    try:
        # Carica speaker attesi
        expected_speakers = get_expected_speakers(args.project_dir)
        if expected_speakers:
            print(f"Speaker attesi: {len(expected_speakers)}")
        
        # Parsing SRT
        print("\nParsing file SRT...")
        subtitles = parse_srt_file(srt_path)
        print(f"   Sottotitoli trovati: {len(subtitles)}")
        
        # Validazione
        print("Validazione sottotitoli...")
        is_valid, errors, warnings, validated_subs = validate_subtitles(subtitles, expected_speakers)
        
        # Report validazione
        print("Generazione report...")
        report_path = save_validation_report(errors, warnings, output_dir)
        
        if not is_valid:
            print(f"\n❌ VALIDAZIONE FALLITA")
            print(f"   Trovati {len(errors)} errori")
            print(f"   Report dettagliato: {report_path}")
            print("\nCorreggere gli errori nel file SRT e rieseguire la fase 6")
            return 1
        
        # Generazione output
        print("Generazione file finali...")
        num_speakers = generate_output_files(validated_subs, output_dir)
        
        # Tempo totale
        elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)
        
        print(f"\n✅ VALIDAZIONE COMPLETATA")
        print(f"   Sottotitoli validati: {len(validated_subs)}")
        print(f"   Speaker trovati: {num_speakers}")

        # Statistiche per speaker
        speaker_stats = get_speaker_stats(validated_subs)
        print(f"   Distribuzione sottotitoli per speaker:")
        for speaker, count in sorted(speaker_stats.items()):
            s = speaker + ":"
            print(f"     - {s:10s} {count:6d}")

        print(f"\n   Errori: {len(errors)}, Avvisi: {len(warnings)}\n")
        print(f"   Report: {report_path}")
        print(f"   File generati in: {output_dir}")
        
        if warnings and not args.quiet:
            print(f"\n⚠️  Sono presenti {len(warnings)} avvisi - controllare il report")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        return 1
    finally:
        if (errors or warnings) and not args.quiet:
            sys.stderr.write("\n")
            with open(report_path, 'r', encoding='utf-8') as report_file:
                sys.stderr.write(report_file.read())

if __name__ == "__main__":
    main()
