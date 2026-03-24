#!/usr/bin/env python3
"""
FASE 5 - Generazione SRT e TXT finali dai risultati Whisper
Crea sottotitoli per speaker e file unificati pronti per l'editing

DESCRIZIONE:
Genera file SRT (sottotitoli) e TXT (testo) a partire dalle trascrizioni Whisper della Fase 4.
Crea file separati per ogni speaker e file combinati con tutti gli speaker interleaved.
I file SRT sono compatibili con software di editing sottotitoli come SubtitleEdit.
"""

import argparse
import json
import os
import sys
import time
import yaml
from collections import defaultdict
from pathlib import Path

# Funzioni comuni
import common_utils

def get_input_dir(project_dir, args):
    """Ottiene la directory di input (trascrizioni Whisper)"""
    if args.input_dir:
        return args.input_dir
    
    # Directory predefinita nella project directory
    return os.path.join(project_dir, "transcripts")


def get_output_dir(project_dir, args):
    """Ottiene la directory di output"""
    if args.output_dir:
        return args.output_dir
    
    # Directory predefinita nella project directory
    return os.path.join(project_dir, "subs")


def format_timestamp_srt(seconds):
    """Formatta timestamp in formato SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


def format_timestamp_readable(seconds):
    """Formatta timestamp in [HH:MM:SS] per TXT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:05.2f}]"


def load_speaker_transcripts(input_dir):
    """Carica tutte le trascrizioni JSON degli speaker"""
    transcripts = {}
    
    # Cerca i file JSON in modo più flessibile
    json_files = list(Path(input_dir).rglob("*_transcript.json"))
    
    if not json_files:
        # Prova pattern alternativo
        json_files = list(Path(input_dir).rglob("*.json"))
        # Filtra i file che non sono di configurazione
        json_files = [f for f in json_files if "config" not in f.name and "stats" not in f.name]
    
    print(f"Trovati {len(json_files)} file JSON")
    
    for json_file in json_files:
        try:
            # Estrai speaker dal nome del file o dalla directory
            speaker = json_file.stem.replace("_transcript", "").replace("_whisper", "")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Verifica la struttura dei dati
            if "segments" in data and data["segments"]:
                transcripts[speaker] = data
            else:
                print(f"  ATTENZIONE: {json_file} non contiene segmenti validi")
                # Prova strutture alternative
                if isinstance(data, list) and len(data) > 0:
                    # Se è una lista di segmenti
                    transcripts[speaker] = {"segments": data}
                    print(f"  Convertito: {speaker} - {len(data)} segmenti (formato lista)")
                
        except Exception as e:
            print(f"  ERRORE nel caricare {json_file}: {e}")
    
    return transcripts


def generate_srt_for_speaker(transcript_data, output_path, use_whisper_segments=False):
    """Genera file SRT per un singolo speaker"""

    if use_whisper_segments:
        # Usa i segmenti Whisper dettagliati
        segments = []
        for seg in transcript_data.get("segments", []):
            segments.extend(seg.get("whisper_segments", []))
    else:
        # Usa i segmenti uniti (default)
        segments = transcript_data.get("segments", [])
    
    if not segments:
        print(f"  ATTENZIONE: Nessun segmento per {output_path}")
        # Crea file vuoto
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        return
    
    segments.sort(key=lambda x: x.get("start", 0))
    
    srt_content = []
    
    for i, seg in enumerate(segments, 1):
        start_time = format_timestamp_srt(seg.get("start", 0))
        end_time = format_timestamp_srt(seg.get("end", seg.get("start", 0) + 1))
        
        # Speaker come prefisso nel testo
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        
        if text:  # Solo se c'è testo
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n[{speaker}] {text}\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_content))


def generate_txt_for_speaker(transcript_data, output_path, with_timestamps=True, use_whisper_segments=False):
    """Genera file TXT per un singolo speaker"""
    if use_whisper_segments:
        # Usa i segmenti Whisper dettagliati
        segments = []
        for seg in transcript_data.get("segments", []):
            segments.extend(seg.get("whisper_segments", []))
    else:
        # Usa i segmenti uniti (default)
        segments = transcript_data.get("segments", [])
    
    if not segments:
        print(f"  ATTENZIONE: Nessun segmento per {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        return
    
    segments.sort(key=lambda x: x.get("start", 0))
    
    txt_content = []
    
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        
        if not text:
            continue
            
        if with_timestamps:
            timestamp = format_timestamp_readable(seg.get("start", 0))
            line = f"{timestamp} [{speaker}] {text}"
        else:
            line = text
        
        txt_content.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_content))


def generate_combined_srt(transcripts, output_path, use_whisper_segments=False):
    """Genera SRT unificato con tutti gli speaker interleaved"""
    all_segments = []

    for speaker, data in transcripts.items():
        if use_whisper_segments:
            # Usa i segmenti Whisper dettagliati
            for seg in data.get("segments", []):
                all_segments.extend(seg.get("whisper_segments", []))
        else:
            # Usa i segmenti uniti (default)
            all_segments.extend(data.get("segments", []))
    
#   for speaker, data in transcripts.items():
#       for seg in data.get("segments", []):
#           seg_copy = seg.copy()
#           seg_copy["speaker"] = speaker  # Assicura che ogni segmento abbia lo speaker
#           all_segments.append(seg_copy)
    
    if not all_segments:
        print(f"  ATTENZIONE: Nessun segmento per {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        return
    
    # Ordina tutti i segmenti per tempo di inizio
    all_segments.sort(key=lambda x: x.get("start", 0))
    
    srt_content = []
    
    for i, seg in enumerate(all_segments, 1):
        start_time = format_timestamp_srt(seg.get("start", 0))
        end_time = format_timestamp_srt(seg.get("end", seg.get("start", 0) + 1))
        
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        
        if text:
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n[{speaker}] {text}\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_content))


def generate_combined_txt(transcripts, output_path, with_timestamps=True, use_whisper_segments=False):
    """Genera TXT unificato con tutti gli speaker"""
    all_segments = []
    
    for speaker, data in transcripts.items():
        for seg in data.get("segments", []):
            seg_copy = seg.copy()
            seg_copy["speaker"] = speaker
            all_segments.append(seg_copy)
    
    if not all_segments:
        print(f"  ATTENZIONE: Nessun segmento per {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        return
    
    all_segments.sort(key=lambda x: x.get("start", 0))
    
    txt_content = []
    
    for seg in all_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        
        if not text:
            continue
            
        if with_timestamps:
            timestamp = format_timestamp_readable(seg.get("start", 0))
            line = f"{timestamp} [{speaker}] {text}"
        else:
            line = f"[{speaker}] {text}"
        
        txt_content.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_content))


def main():
    parser = argparse.ArgumentParser(
        description="FASE 5 - Generazione SRT/TXT finali\n"
                   "Crea sottotitoli per speaker e file unificati pronti per l'editing\n"
                   "I file SRT sono compatibili con SubtitleEdit per la revisione.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--project-dir", required=True, 
        help="Directory del progetto")
    parser.add_argument("--input-dir", 
        help="Directory input trascrizioni (opzionale, default: project-dir/transcripts)")
    parser.add_argument("--output-dir", 
        help="Directory per file finali (opzionale, default: project-dir/subs)")
    parser.add_argument("--force", action="store_true", 
        help="Sovrascrive file esistenti senza chiedere conferma")
    parser.add_argument("--use-whisper-segments", action="store_true", 
        help="Usa i segmenti Whisper dettagliati invece di quelli uniti")

    
    args = parser.parse_args()
    
    # Verifica directory progetto
    if not os.path.exists(args.project_dir):
        print(f"ERRORE: Directory di progetto non trovata: {args.project_dir}")
        return 1

    # Ottieni directory input
    input_dir = get_input_dir(args.project_dir, args)
    if not os.path.exists(input_dir):
        print(f"ERRORE: Directory di input non trovata: {input_dir}")
        print("Assicurati di avere già eseguito fase4.py")
        return 1

    # Ottieni directory output
    output_dir = get_output_dir(args.project_dir, args)
    
    # Verifica sovrascrittura
    if not common_utils.check_existing_output(output_dir, args.force):
        return 1

    start_time = time.time()
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica trascrizioni
    print("Caricamento trascrizioni...")
    transcripts = load_speaker_transcripts(input_dir)
    
    if not transcripts:
        print("Nessuna trascrizione trovata. Assicurati che la Fase 4 sia stata eseguita correttamente.")
        print("Controlla che i file JSON siano nella directory:", input_dir)
        return 1
    
    print(f"Trovati {len(transcripts)} speaker: {list(transcripts.keys())}")
    print(f"Directory di input: {input_dir}")
    print(f"Directory di output: {output_dir}")
    
    # Genera file per ogni speaker
    for speaker, data in transcripts.items():
        
        # SRT per speaker
        srt_path = os.path.join(output_dir, f"{speaker}.srt")
        generate_srt_for_speaker(data, srt_path, args.use_whisper_segments)
        
        # TXT con timestamp
        txt_timestamp_path = os.path.join(output_dir, f"{speaker}_with_timestamps.txt")
        generate_txt_for_speaker(data, txt_timestamp_path, with_timestamps=True, use_whisper_segments=args.use_whisper_segments)
        
        # TXT pulito
        txt_clean_path = os.path.join(output_dir, f"{speaker}_clean.txt")
        generate_txt_for_speaker(data, txt_clean_path, with_timestamps=False)
        
    
    # Genera file combinati
    
    # SRT combinato
    combined_srt_path = os.path.join(output_dir, "podcast_complete.srt")
    generate_combined_srt(transcripts, combined_srt_path, args.use_whisper_segments)
    
    # TXT combinato con timestamp
    combined_txt_path = os.path.join(output_dir, "podcast_with_timestamps.txt")
    generate_combined_txt(transcripts, combined_txt_path, with_timestamps=True)
    
    # TXT combinato pulito
    combined_clean_path = os.path.join(output_dir, "podcast_clean.txt")
    generate_combined_txt(transcripts, combined_clean_path, with_timestamps=False)
    
    print(f"\nCreati file combinati:")
    print(f"  - podcast_complete.srt")
    print(f"  - podcast_with_timestamps.txt") 
    print(f"  - podcast_clean.txt")
    print(f"\nTutti i file generati in: {output_dir}")
    print("\nI file SRT possono essere aperti con SubtitleEdit per la revisione.")
    
    elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)
    return 0


if __name__ == "__main__":
    main()
