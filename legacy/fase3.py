#!/usr/bin/env python3
"""
FASE 3 - Split audio basato su segmenti processati (Fase2)
Estrae frammenti audio per speaker e crea file unici con silenzio/rumore

DESCRIZIONE:
Crea file audio combinati per ogni speaker, estraendo i segmenti identificati nella Fase 2.
Opzionalmente salva i singoli frammenti audio. Utile per creare tracce separate per ogni
oratore o per preparare l'input per la trascrizione Whisper.
"""

import argparse
import json
import os
import sys
import time
import subprocess
import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydub import AudioSegment, silence
from pydub.generators import WhiteNoise

# Funzioni comuni
import common_utils

def validate_input_json(data):
    """Valida la struttura del JSON di input per Fase 3"""
    if not isinstance(data, dict):
        raise ValueError("Il JSON di input deve essere un oggetto")
    
    if 'speakers' not in data:
        raise ValueError("Il JSON di input deve contenere la chiave 'speakers'")
    
    for speaker_id, speaker_data in data['speakers'].items():
        if not isinstance(speaker_data, dict):
            raise ValueError(f"I dati dello speaker '{speaker_id}' non sono un oggetto")
        if 'segments' not in speaker_data:
            raise ValueError(f"Speaker '{speaker_id}' non ha la chiave 'segments'")
        segments = speaker_data['segments']
        if not isinstance(segments, list):
            raise ValueError(f"I segmenti dello speaker '{speaker_id}' non sono una lista")
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                raise ValueError(f"Segmento {i} dello speaker '{speaker_id}' non è un oggetto")
            required_fields = ['start', 'end', 'id']
            for field in required_fields:
                if field not in seg:
                    raise ValueError(f"Segmento {i} dello speaker '{speaker_id}' non ha il campo '{field}'")
            if seg['start'] < 0 or seg['end'] < 0:
                raise ValueError(f"Segmento {i} dello speaker '{speaker_id}' ha tempi negativi")
            if seg['end'] <= seg['start']:
                raise ValueError(f"Segmento {i} dello speaker '{speaker_id}' ha end <= start")


def validate_segment_timing(segments, audio_duration_ms):
    """Valida che i segmenti siano entro la durata dell'audio"""
    warnings = []
    for seg in segments:
        start_ms = seg["start"] * 1000
        end_ms = seg["end"] * 1000
        
        if start_ms > audio_duration_ms:
            warnings.append(f"Segmento {seg['id']} inizia dopo la fine dell'audio ({start_ms/1000:.2f}s > {audio_duration_ms/1000:.2f}s)")
        elif end_ms > audio_duration_ms:
            warnings.append(f"Segmento {seg['id']} finisce dopo la fine dell'audio ({end_ms/1000:.2f}s > {audio_duration_ms/1000:.2f}s)")
    
    return warnings


def get_wav_file_path(project_dir, args, input_json_path):
    """Ottiene il percorso del file WAV (argomento, config, o dal JSON)"""
    if args.input_wav:
        return args.input_wav
    
    # Cerca nel config
    config = common_utils.load_config(project_dir)
    if config and 'paths' in config and 'converted_audio' in config['paths']:
        wav_path = os.path.join(project_dir, config['paths']['converted_audio'])
        if os.path.exists(wav_path):
            return wav_path
    
    # Cerca nel JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        wav_path = data.get("metadata", {}).get("wav_file")
        if wav_path and os.path.exists(wav_path):
            return wav_path
    except:
        pass
    
    return None


def get_output_dir(project_dir, args):
    """Ottiene la directory di output"""
    if args.output_dir:
        return args.output_dir
    
    # Directory predefinita nella project directory
    return os.path.join(project_dir, "combined")


def process_speaker_audio(audio, segments, output_dir, speaker_id, fill_mode, dump_segments=False, verbose=False):
    """Processa tutti i segmenti di uno speaker con gestione errori"""
    speaker_dir = os.path.join(output_dir, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)

    # Validazione timing segmenti
    audio_duration_ms = len(audio)
    timing_warnings = validate_segment_timing(segments, audio_duration_ms)
    if timing_warnings and verbose:
        print(f"  AVVISI timing per {speaker_id}:")
        for warning in timing_warnings:
            print(f"    - {warning}")

    # File per i singoli frammenti (solo se richiesto)
    if dump_segments:
        fragments_dir = os.path.join(speaker_dir, "fragments")
        os.makedirs(fragments_dir, exist_ok=True)

    # Lista per costruire il file unico
    composite_parts = []
    last_end = 0
    processed_fragments = 0

    for i, seg in enumerate(segments):
        try:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            
            # Controlla che il segmento sia valido
            if start_ms >= end_ms:
                print(f"  AVVISO: Segmento {seg['id']} saltato (start >= end)")
                continue
                
            if start_ms >= audio_duration_ms:
                print(f"  AVVISO: Segmento {seg['id']} saltato (inizia dopo fine audio)")
                continue

            # Aggiusta end_ms se supera la durata audio
            end_ms = min(end_ms, audio_duration_ms)
            duration_ms = end_ms - start_ms

            # Estrai il frammento
            fragment = audio[start_ms:end_ms]

            # Salva frammento singolo con padding (se richiesto)
            if dump_segments:
                fragment_filename = f"fragment_{seg['id']:08d}.wav"
                fragment_path = os.path.join(fragments_dir, fragment_filename)
                common_utils.export_audio_segment(fragment, fragment_path)
                if verbose:
                    print(f"  Creato: {speaker_id}/fragments/{fragment_filename}")

            if verbose:
                print(f"  Frammento {seg['id']}: {seg['start']:.2f}s-{seg['end']:.2f}s "
                      f"({duration_ms/1000:.2f}s)")

            # Aggiungi silenzio/rumore prima del frammento (se necessario)
            if start_ms > last_end:
                gap_duration = start_ms - last_end
                filler = common_utils.generate_silence(gap_duration, fill_mode)
                composite_parts.append(filler)

            # Aggiungi il frammento
            composite_parts.append(fragment)
            last_end = end_ms
            processed_fragments += 1

        except Exception as e:
            print(f"  ERRORE durante l'elaborazione del segmento {seg.get('id', 'N/A')}: {e}")
            continue

    # Costruisci il file audio completo per lo speaker
    if composite_parts:
        try:
            composite_audio = composite_parts[0]
            for part in composite_parts[1:]:
                composite_audio += part

            # Aggiungi silenzio finale se necessario
            if last_end < len(audio):
                final_gap = len(audio) - last_end
                filler = common_utils.generate_silence(final_gap, fill_mode)
                composite_audio += filler

            # Salva file composito
            composite_path = os.path.join(speaker_dir, f"{speaker_id}_composite.wav")
            common_utils.export_audio_segment(composite_audio, composite_path)

            print(f"Creato: {speaker_id}/{speaker_id}_composite.wav")

            if verbose:
                print(f"  File composito: {composite_path} ({len(composite_audio)/1000:.2f}s)")

        except Exception as e:
            print(f"  ERRORE durante la creazione del file composito per {speaker_id}: {e}")
            return 0

    return processed_fragments


def main():
    parser = argparse.ArgumentParser(
        description="FASE 3 - Split audio basato su segmenti processati (Fase2)\n"
                   "Crea file audio combinati per ogni speaker con silenzio/rumore tra i segmenti.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--project-dir", required=True, help="Directory del progetto")
    parser.add_argument("--input-json", help="JSON file dalla Fase 2 (opzionale, default: project-dir/fase2_filtered.json)")
    parser.add_argument("--input-wav", help="File audio WAV (opzionale, default: dal config o JSON)")
    parser.add_argument("--output-dir", help="Directory di output (opzionale, default: project-dir/combined)")
    parser.add_argument("--fill-mode", choices=["none", "white", "pink"], default="none",
                       help="Modalità di riempimento silenzi: none=silenzio, white=rumore bianco, pink=rumore rosa")
    parser.add_argument("--speaker", help="Processa solo questo speaker (opzionale)")
    parser.add_argument("--verbose", action="store_true", help="Output dettagliato")
    parser.add_argument("--dump-segments", action="store_true",
                       help="Salva i singoli frammenti audio (default: solo composito)")
    parser.add_argument("--force", action="store_true", help="Sovrascrive file esistenti senza chiedere conferma")
    
    args = parser.parse_args()
    
    # Verifica directory progetto
    if not os.path.exists(args.project_dir):
        print(f"ERRORE: Directory di progetto non trovata: {args.project_dir}")
        sys.exit(1)

    # Ottieni percorso input JSON
    input_json_path = common_utils.get_input_json_path(args.project_dir, args, "fase2_filtered.json")
    if not input_json_path or not os.path.exists(input_json_path):
        print(f"ERRORE: File JSON di input non trovato: {input_json_path}")
        print("Assicurati di avere già eseguito fase1.py e fase2.py")
        sys.exit(1)

    # Ottieni percorso file WAV
    wav_path = get_wav_file_path(args.project_dir, args, input_json_path)
    if not wav_path or not os.path.exists(wav_path):
        print(f"ERRORE: File WAV non trovato: {wav_path}")
        print("Specifica --input-wav o assicurati che il percorso nel JSON/config sia valido")
        sys.exit(1)

    # Ottieni directory output
    output_dir = get_output_dir(args.project_dir, args)

    # Verifica sovrascrittura
    if not common_utils.check_existing_output(output_dir, args.force):
        sys.exit(1)

    # Conteggio del tempo impiegato
    start_time = time.time()

    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica dati Fase 2 con validazione
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validazione struttura JSON
        validate_input_json(data)
        print("✓ Validazione struttura JSON completata")
        
    except Exception as e:
        print(f"ERRORE: Impossibile caricare o validare JSON: {e}")
        sys.exit(1)
    
    # Carica audio
    try:
        audio = AudioSegment.from_wav(wav_path).set_frame_rate(16000).set_channels(1)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare audio: {e}")
        sys.exit(1)
    
    audio_duration_ms = len(audio)
    print(f"Audio caricato: {wav_path}")
    print(f"  Durata: {len(audio)/1000:.2f}s, {audio.frame_rate}Hz, {audio.channels} canale(i)")

    # Validazione speaker richiesti vs disponibili
    available_speakers = list(data['speakers'].keys())
    speakers_to_process = [args.speaker] if args.speaker else available_speakers

    # Controlla se gli speaker richiesti esistono
    if args.speaker and args.speaker not in available_speakers:
        print(f"ERRORE: Speaker '{args.speaker}' non trovato nel JSON")
        print(f"Speaker disponibili: {', '.join(available_speakers)}")
        sys.exit(1)

    print(f"Speaker disponibili: {available_speakers}")
    print(f"Speaker da processare: {speakers_to_process}")
    print(f"Output directory: {output_dir}")
    print(f"Fill mode: {args.fill_mode}")
    
    # Processa speaker
    total_fragments = 0

    print(f"Processing {len(speakers_to_process)} speaker(s)...")
    for speaker_id in speakers_to_process:
        if speaker_id not in data["speakers"]:
            print(f"AVVISO: Speaker '{speaker_id}' non trovato nel JSON")
            continue

        segments = data["speakers"][speaker_id]["segments"]
        if not segments:
            print(f"AVVISO: Nessun segmento per lo speaker '{speaker_id}'")
            continue

        if args.verbose:
            print(f"\nProcessing speaker: {speaker_id} ({len(segments)} segmenti)")

        try:
            fragments_count = process_speaker_audio(
                audio, segments, output_dir, speaker_id, args.fill_mode, args.dump_segments, args.verbose
            )
            total_fragments += fragments_count
            
            if fragments_count < len(segments):
                print(f"  AVVISO: Processati {fragments_count}/{len(segments)} segmenti per {speaker_id}")
                
        except Exception as e:
            print(f"ERRORE durante l'elaborazione dello speaker {speaker_id}: {e}")
            continue

    # Tempo totale
    elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)

    print(f"\nCompletato! Totale: {total_fragments} frammenti in {output_dir}")
    print(f"Tempo totale: {elapsed_time:.2f} secondi")
    if elapsed_time > 0:
        print(f"Tasso: {total_fragments/elapsed_time:.1f} frammenti/secondo")


if __name__ == "__main__":
    main()
