#!/usr/bin/env python3
"""
FASE 4 - Trascrizione con Whisper dei segmenti processati
Processa ogni segmento individualmente preservando i timestamp originali
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import os
import sys
import time
import tempfile
import yaml
from collections import defaultdict
from pydub import AudioSegment

import torch
import whisper

# Funzioni comuni
import common_utils

# Costante per la durata del padding (in millisecondi)
PADDING_DURATION_MS = 500  # 0.5 secondi

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
        with open(input_json_path) as f:
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
    return os.path.join(project_dir, "transcripts")


def get_whisper_parameters_from_config(project_dir, args):
    """Ottiene i parametri Whisper dal config.yaml se non specificati nella CLI"""
    config = common_utils.load_config(project_dir)
    if not config:
        return args
    
    phase4_config = config.get('phase4', {})
    
    # Parametri con fallback al config
    if not args.model and 'model' in phase4_config:
        args.model = phase4_config['model']
    
    if not args.language and 'language' in phase4_config:
        args.language = phase4_config['language']
    
    if args.temperature is None and 'temperature' in phase4_config:
        args.temperature = phase4_config['temperature']
    
    if args.beam_size is None and 'beam_size' in phase4_config:
        args.beam_size = phase4_config['beam_size']
    
    if args.best_of is None and 'best_of' in phase4_config:
        args.best_of = phase4_config['best_of']
    
    if args.no_speech_threshold is None and 'no_speech_threshold' in phase4_config:
        args.no_speech_threshold = phase4_config['no_speech_threshold']
    
    if args.compression_ratio_threshold is None and 'compression_ratio_threshold' in phase4_config:
        args.compression_ratio_threshold = phase4_config['compression_ratio_threshold']
    
    return args



# Modifica la funzione create_padding_audio per usare common_utils
def create_padding_audio(frame_rate=16000, channels=1):
    """Crea un segmento di rumore rosa da riutilizzare per tutti i padding"""
    padding = common_utils.generate_silence(PADDING_DURATION_MS, "pink")
    # Converti al sample rate e canali desiderati per coerenza
    return padding.set_frame_rate(frame_rate).set_channels(channels)



def transcribe_segment(speaker, audio_segment, model, whisper_args, original_start_time, padding_audio=None):
    """
    Trascrive un singolo segmento audio con Whisper e mappa i timestamp all'audio originale
    Comportamento: sempre merge dei segmenti Whisper, ma conserva l'elenco originale
    """
    # Aggiungi padding prima e dopo il segmento se specificato
    if padding_audio is not None:
        audio_segment = padding_audio + audio_segment + padding_audio
        padding_duration = PADDING_DURATION_MS / 1000.0  # Converti in secondi
    else:
        padding_duration = 0.0

    # Salva temporaneamente il segmento audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        audio_segment.export(temp_path, format="wav")
    
    try:
        # Trascrizione con Whisper
        whisper_args = whisper_args.copy()
        result = model.transcribe(temp_path, **whisper_args)
        
        # Conserva i segmenti Whisper originali (con timestamp corretti)
        original_whisper_segments = []
        original_duration = (len(audio_segment) / 1000.0) - (2 * padding_duration)  # Durata senza padding
        
        for seg in result.get("segments", []):
            # Mappa i timestamp al tempo originale, sottraendo il padding iniziale
            seg_start = seg["start"] - padding_duration
            seg_end = seg["end"] - padding_duration
            
            # Evita timestamp negativi
            seg_start = max(0, seg_start)
            seg_end = max(0, seg_end)
            
            # Se il segmento finisce dopo la fine del segmento originale (dentro il padding finale),
            # troncalo alla fine del segmento originale
            if seg_end > original_duration:
                seg_end = original_duration
            
            # Salta segmenti che sono completamente nel padding finale
            if seg_start >= original_duration:
                continue
                
            # Calcola i timestamp assoluti nell'audio originale
            absolute_start = original_start_time + seg_start
            absolute_end = original_start_time + seg_end
            
            # Salta segmenti con durata nulla o negativa
            if absolute_end <= absolute_start:
                continue
            
            original_whisper_segments.append({
                "speaker": speaker,
                "start": absolute_start,
                "end": absolute_end,
                "text": seg["text"].strip(),
                "confidence": seg.get("confidence", 0.0),
                "words": seg.get("words", [])
            })
        
        # Unisci sempre i segmenti Whisper in uno solo con i timestamp originali
        full_text = " ".join(seg["text"].strip() for seg in result.get("segments", []))
        
        whisper_segments = [{
            "start": original_start_time,
            "end": original_start_time + original_duration,  # Usa la durata originale senza padding
            "text": full_text,
            "confidence": None,
            "words": [],
            "whisper_segments": original_whisper_segments  # Aggiungi i segmenti originali
        }]
        
        return {
            "text": result["text"].strip(),
            "segments": whisper_segments,
            "language": result.get("language", whisper_args.get("language", "unknown"))
        }
        
    except Exception as e:
        print(f"    ERRORE nella trascrizione: {e}")
        return {
            "text": "",
            "segments": [],
            "error": str(e)
        }
    finally:
        # Pulizia file temporaneo
        if os.path.exists(temp_path):
            os.unlink(temp_path)



def format_timestamp(seconds):
    """Formatta timestamp in [HH:MM:SS.mmm]"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:06.3f}]"


def save_speaker_transcripts(whisper_segments, speaker, output_dir, audio_duration):
    """Salva trascrizioni per un singolo speaker nel formato compatibile con fase5.py"""
    speaker_dir = os.path.join(output_dir, speaker)
    os.makedirs(speaker_dir, exist_ok=True)
    
    # Prepara i segmenti nel formato richiesto da fase5.py
    segments_formatted = []
    for seg in whisper_segments:
        segment_data = {
            "speaker": speaker,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "confidence": seg.get("confidence", 0.0),
            "whisper_segments": seg.get("whisper_segments", [])
        }
        segments_formatted.append(segment_data)
    
    # Crea la struttura dati principale nel formato richiesto
    main_data = {
        "speaker": speaker,
        "audio_duration": audio_duration,
        "segments": segments_formatted
    }
    
    # 1. JSON principale nel formato compatibile con fase5.py
    json_path = os.path.join(speaker_dir, f"{speaker}_transcript.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)
    
    # 2. Testo con timestamp (da segmenti Whisper)
    txt_with_timestamps = []
    current_pos = 0.0
    
    for seg in whisper_segments:
        # Aggiungi silenzio/pause tra segmenti
        if seg["start"] > current_pos + 0.1:  # Soglia piccola pausa
            gap = seg["start"] - current_pos
            txt_with_timestamps.append(f"\n[PAUSA: {gap:.1f}s]\n")
        
        timestamp = format_timestamp(seg["start"])
        txt_with_timestamps.append(f"{timestamp} {seg['text']}")
        current_pos = seg["end"]
    
    txt_path = os.path.join(speaker_dir, f"{speaker}_transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_with_timestamps))
    
    # 3. Testo pulito (solo testo)
    clean_text = " ".join(seg["text"] for seg in whisper_segments)
    clean_path = os.path.join(speaker_dir, f"{speaker}_transcript_clean.txt")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
    
    return json_path, txt_path, clean_path


def main():
    parser = argparse.ArgumentParser(description="FASE 4 - Trascrizione con Whisper (segmenti separati)")
    
    # Input/Output
    parser.add_argument("--project-dir", required=True, help="Directory del progetto")
    parser.add_argument("--input-json-fase2", help="JSON file dalla Fase 2 (opzionale, default: project-dir/fase2_filtered.json)")
    parser.add_argument("--input-wav", help="File audio WAV (opzionale, default: dal config o JSON)")
    parser.add_argument("--output-dir", help="Directory di output (opzionale, default: project-dir/transcripts)")
    parser.add_argument("--speaker", help="Processa solo questo speaker")
    
    # Parametri Whisper
    parser.add_argument("--model", help="Modello Whisper (default: dal config o large-v3)")
    parser.add_argument("--language", help="Lingua audio (default: dal config o de)")
    parser.add_argument("--temperature", type=float, help="Temperature Whisper")
    parser.add_argument("--beam_size", type=int, help="Beam size Whisper")
    parser.add_argument("--best_of", type=int, help="Best of Whisper")
    parser.add_argument("--no_speech_threshold", type=float, help="No speech threshold")
    parser.add_argument("--compression_ratio_threshold", type=float, help="Compression ratio threshold")
    parser.add_argument("--word_timestamps", action="store_true", help="Includi timestamp a livello di parola")
    
    parser.add_argument("--force", action="store_true", help="Sovrascrive file esistenti senza chiedere conferma")
    
    args = parser.parse_args()
    
    # Verifica directory progetto
    if not os.path.exists(args.project_dir):
        print(f"ERRORE: Directory di progetto non trovata: {args.project_dir}")
        return 1

    # Ottieni parametri Whisper dal config
    args = get_whisper_parameters_from_config(args.project_dir, args)
    
    # Ottieni percorso input JSON
    input_json_path = common_utils.get_input_json_path(args.project_dir, args)
    if not input_json_path or not os.path.exists(input_json_path):
        print(f"ERRORE: File JSON di Fase 2 non trovato: {input_json_path}")
        print("Assicurati di avere già eseguito fase2.py")
        return 1
    
    # Ottieni percorso file WAV
    wav_path = get_wav_file_path(args.project_dir, args, input_json_path)
    if not wav_path or not os.path.exists(wav_path):
        print(f"ERRORE: File WAV non trovato: {wav_path}")
        print("Specifica --input-wav o assicurati che il percorso nel JSON/config sia valido")
        return 1
    
    # Ottieni directory output
    output_dir = get_output_dir(args.project_dir, args)
    
    # Verifica sovrascrittura
    if not common_utils.check_existing_output(output_dir, args.force):
        return 1
    
    # Conteggio del tempo impiegato
    start_time = time.time()

    # Carica dati Fase 2
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            fase2_data = json.load(f)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare JSON Fase 2: {e}")
        return 1
    
    # Prepara parametri Whisper
    whisper_args = {
        "language": args.language or "de",
        "word_timestamps": args.word_timestamps
    }
    
    # Aggiungi parametri opzionali
    optional_args = ["temperature", "beam_size", "best_of", 
                    "no_speech_threshold", "compression_ratio_threshold"]
    for arg_name in optional_args:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            whisper_args[arg_name] = arg_value
    
    # Carica modello Whisper
    model_name = args.model or "large-v3"
    print(f"Caricamento modello Whisper: {model_name}")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare modello Whisper: {e}")
        return 1
    
    # Determina speaker da processare
    speakers_to_process = [args.speaker] if args.speaker else list(fase2_data["speakers"].keys())
    
    print(f"Elaborazione di {len(speakers_to_process)} speaker(s)...")
    print(f"Parametri Whisper: { {k: v for k, v in whisper_args.items() if k != 'verbose'} }")
    print(f"Output directory: {output_dir}")
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica audio completo
    print("Caricamento file audio...")
    try:
        full_audio = AudioSegment.from_wav(wav_path)
        audio_duration = len(full_audio) / 1000.0
        print(f"Durata audio: {audio_duration:.1f}s")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare file audio: {e}")
        return 1
    
    # Crea il segmento di padding (rumore rosa) da riutilizzare
    print("Preparazione padding audio (rumore rosa)...")
    padding_audio = create_padding_audio(
        frame_rate=full_audio.frame_rate,
        channels=full_audio.channels
    )
    
    # Processa ogni speaker
    for speaker in speakers_to_process:
        if speaker not in fase2_data["speakers"]:
            print(f"AVVISO: Speaker '{speaker}' non trovato, salto")
            continue
        
        segments = fase2_data["speakers"][speaker]["segments"]
        if not segments:
            print(f"AVVISO: Nessun segmento per speaker '{speaker}', salto")
            continue
        
        print(f"\n=== PROCESSING SPEAKER: {speaker} ===")
        print(f"Segmenti da processare: {len(segments)}")
        
        all_whisper_segments = []
        
        # Processa ogni segmento individualmente
        for i, seg in enumerate(segments):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            duration = seg["end"] - seg["start"]

            # Estrai il segmento audio originale
            try:
                audio_segment = full_audio[start_ms:end_ms]

                # Trascrizione con padding
                result = transcribe_segment(seg["speaker"], audio_segment, model, whisper_args,
                                  seg["start"], padding_audio)

                # Prepara anteprima testo per display
                if "error" in result:
                    anteprima = f"ERRORE: {result['error'][:40]}"
                else:
                    anteprima = result['text'][:60].replace('\n', ' ') + ('...' if len(result['text']) > 60 else '')

                # Display su singola riga che si aggiorna completamente
                terminal_width = 80  # Larghezza tipica del terminale
                progress_text = f"Segmento {i+1}/{len(segments)}: {anteprima}"
                # Riempie con spazi per cancellare la riga precedente
                padded_text = progress_text.ljust(terminal_width)
                print(f"\r{padded_text}", end="", flush=True)

                # Aggiungi segmenti Whisper alla lista globale
                for whisper_seg in result["segments"]:
                    whisper_seg["speaker"] = speaker
                    all_whisper_segments.append(whisper_seg)

            except Exception as e:
                error_text = f"Segmento {i+1}/{len(segments)}: ERRORE - {str(e)[:40]}"
                padded_error = error_text.ljust(80)
                print(f"\r{padded_error}", end="", flush=True)
 
        # Vai a capo dopo aver completato tutti i segmenti di questo speaker
        print()
        
        # Ordina i segmenti Whisper per tempo
        all_whisper_segments.sort(key=lambda x: x["start"])
        
        # Salva file per speaker nel formato compatibile con fase5.py
        json_path, txt_path, clean_path = save_speaker_transcripts(
            all_whisper_segments, speaker, output_dir, audio_duration
        )
        
        print(f"Trascrizioni salvate:")
        print(f"  - JSON: {json_path}")
        print(f"  - Con timestamp: {txt_path}")
        print(f"  - Testo pulito: {clean_path}")
    
    elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)
    print(f"\n=== FASE 4 COMPLETATA ===")
    print(f"Tempo totale: {elapsed_time:.2f} secondi")
    
    return 0


if __name__ == "__main__":
    main()
