#!/usr/bin/env python3
"""
FASE 0 - Diarizzazione a blocchi con PyAnnote 4.0
Split audio in blocchi fissi rispettando i segmenti PyAnnote
Compatibile con pyannote/speaker-diarization-community-1
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message=".*torch.*")


import argparse
import json
import subprocess
import os
import sys
import time
import hashlib
import re
import yaml
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment
from pydub import AudioSegment, silence
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# Funzioni comuni
import common_utils

def load_or_create_config(project_dir, args):
    """Carica config esistente o crea nuovo con parametri da CLI"""
    config_path = os.path.join(project_dir, "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Config caricato: {config_path}")
    else:
        config = {
            'project': {
                'name': os.path.basename(project_dir),
                'created': time.strftime("%Y-%m-%d")
            },
            'paths': {
                'converted_audio': "audio_converted.wav"
            }
        }
        print(f"Config creato: {config_path}")

    # Assicurati che esista la sezione phase0
    if not config.get('phase0'):
        config['phase0'] = {}

    # Se viene passato --num-blocks, rimuovi block_duration dalla config e viceversa
    if args.num_blocks is not None:
        if 'block_duration' in config['phase0']:
            del config['phase0']['block_duration']
    elif args.block_duration: 
        if 'num_blocks' in config['phase0']:
            del config['phase0']['num_blocks']

    # Aggiorna config con parametri CLI (override)
    if args.token:
        config['phase0']['token'] = args.token

    # Parametri opzionali - solo se forniti via CLI
    if args.min_speakers is not None:
        config['phase0']['min_speakers'] = args.min_speakers
    if args.max_speakers is not None:
        config['phase0']['max_speakers'] = args.max_speakers
    if args.num_speakers is not None:
        config['phase0']['num_speakers'] = args.num_speakers
    if args.block_duration:
        config['phase0']['block_duration'] = args.block_duration
    if args.num_blocks is not None:
        config['phase0']['num_blocks'] = args.num_blocks
    if args.sample_duration:
        config['phase0']['sample_duration'] = args.sample_duration
    if args.residual_threshold:
        config['phase0']['residual_threshold'] = args.residual_threshold

    # Salva config aggiornato
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    return config

def check_required_params(config, args):
    """Verifica parametri obbligatori per fase0"""
    phase0_config = config.get('phase0', {})
    
    # Ottiene hoken HF da var. di environment, file di configurazione o parametro
    mytoken = get_hf_token(args, config)
    if not mytoken:
        print("ERRORE: Token Hugging Face obbligatorio")
        print("Specifica --token o aggiungilo al config.yaml nella directory di progetto")
        print("oppure imposta la variabile d'ambiente HUGGINGFACE_HUB_TOKEN o HF_TOKEN")
        return False
    
    # Input è obbligatorio
    if not args.input:
        print("ERRORE: --input obbligatorio")
        return False
    
    return True


def get_hf_token(args, config):
    """Ottiene il token Hugging Face con priorità: CLI > Config > Env Var"""
    # 1. Priorità alla command line
    if args.token:
        return args.token

    # 2. Cerca nel config
    phase0_config = config.get('phase0', {})
    if phase0_config.get('token'):
        return phase0_config['token']

    # 3. Cerca in variabili d'ambiente (convenzionali per Hugging Face)
    env_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    if env_token:
        return env_token

    return None

def extract_audio_segment(input_path, output_path, start_time, end_time):
    """Estrai segmento audio con ffmpeg"""
    cmd = f"ffmpeg -i '{input_path}' -ss {start_time} -to {end_time} -acodec pcm_s16le -ar 16000 -ac 1 -y '{output_path}' 2>/dev/null"
    subprocess.run(cmd, shell=True, check=True)
    return output_path


def process_audio_block(audio_path, start_time, max_duration, total_duration, pipeline, device, pipeline_params, id_offset=0, exclusive_mode=False):
    """
    Processa un blocco di audio di massimo max_duration secondi
    Versione semplificata - estrae segmenti e embeddings in modo più efficiente
    """
    temp_audio_path = f"temp_block_{int(time.time())}_{os.getpid()}.wav"
    end_time_target = start_time + max_duration

    try:
        # Estrai il blocco audio
        extract_audio_segment(audio_path, temp_audio_path, start_time, end_time_target)

        # Diarizzazione con progress bar
        with ProgressHook() as hook:
            diarization = pipeline(temp_audio_path, hook=hook, **pipeline_params)

        # Seleziona il tipo di diarizzazione in base alla modalità
        if exclusive_mode and hasattr(diarization, 'exclusive_speaker_diarization'):
            diarization_source = diarization.exclusive_speaker_diarization
        elif hasattr(diarization, 'speaker_diarization'):
            diarization_source = diarization.speaker_diarization
        else:
            diarization_source = diarization

        # UNICO LOOP: estrai segmenti e raccogli speaker
        segments_data = []
        speakers_in_segments = set()

        for turn, speaker in diarization_source:
            absolute_start = start_time + turn.start
            absolute_end = start_time + turn.end

            segments_data.append({
                "id": len(segments_data) + 1 + id_offset,
                "start": round(absolute_start, 8),
                "end": round(absolute_end, 8),
                "duration": round(absolute_end - absolute_start, 8),
                "speaker": speaker,
                "confidence": 0.7,
                "type": "normal",
                "overlaps_with": [],
                "includes": [],
                "included_in": None
            })
            speakers_in_segments.add(speaker)

        # Estrazione embeddings - SOLO SE ABBIAMO SEGMENTI
        speaker_embeddings_dict = {}

        if segments_data and hasattr(diarization, 'speaker_embeddings') and diarization.speaker_embeddings is not None:
            try:
                sorted_speakers = sorted(list(speakers_in_segments))

                if hasattr(diarization.speaker_embeddings, '__iter__'):
                    embeddings_list = list(diarization.speaker_embeddings)

                    if len(embeddings_list) >= len(sorted_speakers):
                        for i, speaker in enumerate(sorted_speakers):
                            if i < len(embeddings_list):
                                speaker_embeddings_dict[speaker] = embeddings_list[i].tolist() if hasattr(embeddings_list[i], 'tolist') else embeddings_list[i]
                    else:
                        print(f"Warning: Meno embeddings ({len(embeddings_list)}) che speaker ({len(sorted_speakers)}) nel blocco")

            except Exception as e:
                print(f"Errore nell'estrazione embeddings: {e}")

        # [MANTIENI IL RESTO DEL CODICE PER RELAZIONI TRA SEGMENTI E GESTIONE BLOCCHI...]
        # Rileva relazioni tra segmenti (sovrapposizioni e inclusioni)
        segments_data.sort(key=lambda x: x["start"])

        for i in range(len(segments_data)):
            seg1 = segments_data[i]
            # ... codice per relazioni tra segmenti ...

        # Check ultimo blocco e calcolo next_start
        remaining_time = total_duration - start_time
        is_final_block = remaining_time < (max_duration / 3)

        if is_final_block or len(segments_data) <= 1:
            segments_to_keep = segments_data
            next_start = segments_data[-1]["end"] if segments_data else start_time + 30.0
        else:
            segments_to_keep = segments_data[:-1]
            next_start = segments_data[-1]["start"]

        # Calcola il prossimo offset ID
        max_id = max(seg["id"] for seg in segments_to_keep) if segments_to_keep else id_offset
        next_id_offset = max_id + 1

        return segments_to_keep, next_start, next_id_offset, speaker_embeddings_dict

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def extract_speaker_samples(audio_path, segments, block_dir, sample_duration=60):
    """Estrai campioni audio per ogni speaker - versione semplificata e robusta"""
    speaker_samples = {}
    
    for speaker in set(seg['speaker'] for seg in segments):
        speaker_segments = [s for s in segments if s['speaker'] == speaker]
        
        if not speaker_segments:
            continue
            
        # Ordina i segmenti per durata (dal più lungo al più corto)
        speaker_segments.sort(key=lambda x: (-x['duration'], x['start']))
        
        selected_segments = []
        current_duration = 0
        gap_duration = 0.2  # Secondi di pausa tra segmenti
        
        # Seleziona segmenti fino a raggiungere la durata target
        for seg in speaker_segments:
            if current_duration >= sample_duration:
                break
                
            # Calcola quanto possiamo prendere da questo segmento
            max_segment_duration = 10.0  # Massimo 10 secondi per segmento
            segment_available = min(seg['duration'], max_segment_duration)
            needed_duration = sample_duration - current_duration
            
            # Quanto prendere da questo segmento
            take_duration = min(segment_available, needed_duration)
            
            if take_duration > 1.0:  # Solo se il segmento è abbastanza lungo
                # Prendi la parte centrale del segmento
                segment_mid = seg['start'] + (seg['duration'] / 2)
                segment_start = segment_mid - (take_duration / 2)
                segment_end = segment_start + take_duration
                
                # Aggiusta se fuori dai limiti del segmento originale
                if segment_start < seg['start']:
                    segment_start = seg['start']
                    segment_end = segment_start + take_duration
                elif segment_end > seg['end']:
                    segment_end = seg['end']
                    segment_start = segment_end - take_duration
                
                selected_segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'duration': take_duration
                })
                current_duration += take_duration
        
        # Se non abbiamo abbastanza materiale, usa l'approccio originale come fallback
        if not selected_segments:
            # Fallback: prendi il segmento più lungo (max 60s)
            longest_segment = speaker_segments[0]
            sample_start = longest_segment['start']
            sample_end = min(longest_segment['end'], sample_start + sample_duration)
            
            selected_segments.append({
                'start': sample_start,
                'end': sample_end,
                'duration': sample_end - sample_start
            })
        
        # APPROCCIO SEMPLIFICATO: estrai direttamente il sample finale
        sample_filename = f"{speaker}_sample.wav"
        sample_path = os.path.join(block_dir, sample_filename)
        
        if len(selected_segments) == 1:
            # Caso semplice: un solo segmento
            seg_info = selected_segments[0]
            cmd = (
                f"ffmpeg -y -i '{audio_path}' -ss {seg_info['start']} -to {seg_info['end']} "
                f"-acodec pcm_s16le -ar 16000 -ac 1 '{sample_path}' 2>/dev/null"
            )
        else:
            # Caso multiplo: crea filtro complex per concatenare
            filter_complex = ""
            inputs = ""
            concat_inputs = ""
            
            for i, seg_info in enumerate(selected_segments):
                # Aggiungi input per ogni segmento
                filter_complex += f"[{i}:a]"
                inputs += f" -ss {seg_info['start']} -to {seg_info['end']} -i '{audio_path}'"
                concat_inputs += f"[{i}:a]"
            
            # Aggiungi pause di 0.2s tra i segmenti
            gap_filter = ""
            for i in range(len(selected_segments) - 1):
                gap_filter += f"anullsrc=channel_layout=mono:sample_rate=16000:duration=0.2[gap{i}];"
                concat_inputs += f"[gap{i}]"
            
            concat_inputs += f"concat=n={len(selected_segments) * 2 - 1}:v=0:a=1[outa]"
            
            cmd = (
                f"ffmpeg -y{inputs} -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000 "
                f"-filter_complex \"{gap_filter}{concat_inputs}\" "
                f"-map '[outa]' -acodec pcm_s16le -ar 16000 -ac 1 -t {current_duration} '{sample_path}' 2>/dev/null"
            )
        
        try:
            # Esegui il comando ffmpeg
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True)
            if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
                speaker_samples[speaker] = sample_path
            else:
                print(f"Warning: Sample non creato per {speaker}")
                
        except subprocess.CalledProcessError as e:
            print(f"Errore ffmpeg per {speaker}: {e}")
            # Fallback ultra-semplice: primo segmento
            try:
                seg_info = selected_segments[0]
                cmd_fallback = (
                    f"ffmpeg -y -i '{audio_path}' -ss {seg_info['start']} -to {seg_info['end']} "
                    f"-acodec pcm_s16le -ar 16000 -ac 1 '{sample_path}' 2>/dev/null"
                )
                subprocess.run(cmd_fallback, shell=True, check=True)
                if os.path.exists(sample_path):
                    speaker_samples[speaker] = sample_path
            except:
                print(f"Fallback fallito per {speaker}")
    
    return speaker_samples


def generate_speaker_map(blocks_data, output_dir):
    """Genera file di mappatura speaker con istruzioni migliorate"""
    map_path = os.path.join(output_dir, "speakers_map.txt")
    
    with open(map_path, 'w', encoding='utf-8') as f:
        f.write("######################################################################\n")
        f.write("# MAPPA SPEAKER - ISTRUZIONI\n")
        f.write("######################################################################\n")
        f.write("# PyAnnote riconosce gli speaker indipendentemente per ogni blocco.\n")
        f.write("# Questa mappa serve per associarli a quelli reali in modo univoco.\n")
        f.write("# \n")
        f.write("# FORMATO: BLOCK_BB.SPEAKER_NN => NOME_UNIVOCO\n")
        f.write("# \n")
        f.write("# ESEMPI:\n")
        f.write("#   BLOCK_03.SPEAKER_00 => INTERVISTATORE_1\n")
        f.write("#   BLOCK_03.SPEAKER_01 => SIGLA_MUSICALE\n")
        f.write("#   BLOCK_03.SPEAKER_02 => INTERVISTATO_1\n")
        f.write("#   BLOCK_03.SPEAKER_03 => OSPITE_SPECIALE\n")
        f.write("# \n")
        f.write("# CONSIGLI:\n")
        f.write("# - Usa nomi che siano sensati per te, senza spazi. \n")
        f.write("#   Esempio: INTERV_1, Ospite, SIGLA, SCARTA\n")
        f.write("# - Lo stesso speaker in blocchi diversi deve avere lo stesso nome\n")
        f.write("# - Ascolta i campioni audio per identificare gli speaker\n")
        f.write("# - Tutte le voci DEVONO essere mappate per proseguire alla fase 2\n")
        f.write("# \n")
        f.write("# NOTA: I campioni audio sono disponibili nelle cartelle BLOCK_XX/\n")
        f.write("######################################################################\n\n")
        
        # Per il primo blocco, suggerisci nomi automatici
        first_block_speakers = []
        if blocks_data:
            first_block_speakers = sorted(list(set(
                seg['speaker'] for seg in blocks_data[0]['segments']
            )))
        
        global_speaker_map = {}
        for i, speaker in enumerate(first_block_speakers):
            global_speaker_map[speaker] = f"SPEAKER_{chr(65 + i)}"  # A, B, C, ...
        
        for block_id, block_data in enumerate(blocks_data):
            block_start = block_data['metadata']['start_time']
            block_end = block_data['metadata']['end_time']
            
            f.write(f"# BLOCCO {block_id:02d} ({block_start:.1f}s - {block_end:.1f}s)\n")
            
            speakers = sorted(list(set(seg['speaker'] for seg in block_data['segments'])))
            
            for speaker in speakers:
                block_speaker_id = f"BLOCK_{block_id:02d}.{speaker}"
                
                if block_id == 0:
                    # Primo blocco: suggerisci nomi automatici
                    if speaker in global_speaker_map:
                        f.write(f"{block_speaker_id} => {global_speaker_map[speaker]}\n")
                    else:
                        f.write(f"{block_speaker_id} => \n")
                else:
                    # Blocchi successivi: lascia vuoto per scelta manuale
                    f.write(f"{block_speaker_id} => \n")
            
            f.write("\n")
        
        f.write("# DOPO AVER COMPILATO, SALVA IL FILE E PROSEGUI CON FASE 1\n")
    
    return map_path


def generate_block_report(block_data, block_dir):
    """Genera report statistiche per un blocco (con dati completi)"""
    report_path = os.path.join(block_dir, f"block_{block_data['metadata']['block_id']:02d}_report.txt")

    segments = block_data['segments']
    speakers = set(seg['speaker'] for seg in segments)

    report_lines = []
    report_lines.append("=== REPORT ANALISI COMPLETO ===")
    report_lines.append(f"Speaker rilevati: {len(speakers)}")
    report_lines.append(f"Segmenti totali: {len(segments)}")

    # Lunghezze segmenti
    report_lines.append("\nLunghezze segmenti:")
    duration_ranges = [
        ("<5s", 0, 5),
        ("5-20s", 5, 20),
        ("20s-1m", 20, 60),
        ("1m-2m", 60, 120),
        (">2m", 120, float('inf'))
    ]

    duration_bins = {name: 0 for name, _, _ in duration_ranges}
    for seg in segments:
        for name, min_val, max_val in duration_ranges:
            if min_val <= seg['duration'] < max_val:
                duration_bins[name] += 1
                break

    for bin_name in [r[0] for r in duration_ranges]:
        report_lines.append(f"- {bin_name}: {duration_bins[bin_name]} segmenti")

    # Tipi di segmento (ora con dati reali)
    report_lines.append("\nTipi di segmento:")
    segment_types = defaultdict(int)
    for seg in segments:
        segment_types[seg['type']] += 1

    for seg_type, count in segment_types.items():
        report_lines.append(f"- {seg_type}: {count} segmenti")

    # Statistiche per speaker (completa)
    report_lines.append("\n=== STATISTICHE PER SPEAKER ===")

    for speaker in sorted(speakers):
        speaker_segments = [s for s in segments if s['speaker'] == speaker]
        durations = [s['duration'] for s in speaker_segments]
        confidences = [s['confidence'] for s in speaker_segments]
        types = [s['type'] for s in speaker_segments]

        total_duration = sum(durations)
        avg_duration = np.mean(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        min_conf = min(confidences) if confidences else 0
        max_conf = max(confidences) if confidences else 0

        # Frammentazione
        if total_duration > 0:
            fragmentation = (len(speaker_segments) / total_duration) * 60
        else:
            fragmentation = 0

        # Pause tra interventi
        turn_gaps = []
        segs_sorted = sorted(speaker_segments, key=lambda x: x['start'])
        for i in range(1, len(segs_sorted)):
            gap = segs_sorted[i]['start'] - segs_sorted[i-1]['end']
            if gap > 0:
                turn_gaps.append(gap)
        avg_gap = np.mean(turn_gaps) if turn_gaps else 0

        # Segmenti sovrapposti
        overlapped_count = sum(1 for s in speaker_segments if s['type'] == 'overlapped')
        overlapped_pct = (overlapped_count / len(speaker_segments) * 100) if speaker_segments else 0

        # Distribuzione tipi
        type_counts = defaultdict(int)
        for seg_type in types:
            type_counts[seg_type] += 1

        report_lines.append(f"\nSpeaker: {speaker}")
        report_lines.append(f"- Segmenti: {len(speaker_segments)}")
        report_lines.append(f"- Durata minima: {min_duration:.2f}s")
        report_lines.append(f"- Durata massima: {max_duration:.2f}s")
        report_lines.append(f"- Durata media: {avg_duration:.2f}s")
        report_lines.append(f"- Durata totale: {total_duration:.2f}s")
        report_lines.append(f"- Confidenza media: {avg_confidence:.3f}")
        report_lines.append(f"- Confidenza minima: {min_conf:.3f}")
        report_lines.append(f"- Confidenza massima: {max_conf:.3f}")
        report_lines.append(f"- Frammentazione: {fragmentation:.1f} seg/min")
        report_lines.append(f"- Segmenti sovrapposti: {overlapped_pct:.1f}%")
        report_lines.append(f"- Pausa media tra interventi: {avg_gap:.1f}s")
        report_lines.append(f"- Tipi di segmento:")
        for t, c in type_counts.items():
            report_lines.append(f"  * {t}: {c}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    return report_path


def extract_embeddings_from_diarization(blocks_data):
    """
    Estrae embedding direttamente dai risultati di diarizzazione
    invece di ricalcolarli con un modello separato
    """
    embeddings_dict = {}
    
    for block_data in blocks_data:
        block_id = block_data['metadata']['block_id']
        
        # Gli embeddings sono già nel block_data se li abbiamo salvati durante la diarizzazione
        if 'speaker_embeddings' in block_data:
            for speaker, embedding in block_data['speaker_embeddings'].items():
                embeddings_dict[(str(block_id), speaker)] = embedding
    
    return embeddings_dict


def cluster_speakers(embeddings_dict, eps=0.3, min_samples=1):
    """
    Clusterizza speaker usando DBSCAN sugli embedding
    Restituisce etichette di cluster per ogni speaker
    """
    if not embeddings_dict:
        return {}

    speakers = list(embeddings_dict.keys())
    # Usa la funzione helper da common_utils per convertire in array numpy
    embeddings = np.array([common_utils.ensure_numpy_array(embeddings_dict[speaker]) for speaker in speakers])

    # Clusterizzazione DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    # Mappa speaker -> cluster
    speaker_clusters = {}
    for i, speaker in enumerate(speakers):
        speaker_clusters[speaker] = labels[i]

    return speaker_clusters


def generate_auto_mapping(embeddings_dict, eps=0.3, min_samples=1):
    """
    Genera mappatura automatica basata su clustering
    Assegna nomi generici: SPEAKER_A, SPEAKER_B, ...
    """
    if not embeddings_dict:
        return {}

    # Clusterizzazione per identificare speaker unici
    speaker_clusters = cluster_speakers(embeddings_dict, eps, min_samples)

    # Raggruppa speaker per cluster
    cluster_groups = defaultdict(list)
    for speaker, cluster_id in speaker_clusters.items():
        cluster_groups[cluster_id].append(speaker)

    # Assegna nomi generici
    suggested_mapping = {}
    cluster_names = {}

    # Ordina i cluster per id e assegna nomi in ordine
    sorted_clusters = sorted([c for c in cluster_groups.keys() if c != -1])
    outlier_clusters = [c for c in cluster_groups.keys() if c == -1]

    # Assegna nomi ai cluster validi
    for idx, cluster_id in enumerate(sorted_clusters):
        cluster_name = f"SPEAKER_{chr(65 + idx)}"  # A, B, C, ...
        cluster_names[cluster_id] = cluster_name

        # Applica a tutti gli speaker del cluster
        for speaker in cluster_groups[cluster_id]:
            block_speaker_id = f"BLOCK_{speaker[0]}.{speaker[1]}"
            suggested_mapping[block_speaker_id] = cluster_name

    # Gestisci outliers (cluster_id = -1)
    for cluster_id in outlier_clusters:
        for speaker in cluster_groups[cluster_id]:
            block_id, speaker_id = speaker
            suggested_name = f"SPEAKER_OUTLIER_{block_id}_{speaker_id}"
            suggested_mapping[f"BLOCK_{block_id}.{speaker_id}"] = suggested_name

    return suggested_mapping, cluster_groups


def save_auto_mapping(project_dir, suggested_mapping, cluster_groups):
    """Salva mappatura automatica in speakers_map.txt e suggested_speakers_map.txt"""

    # File mappatura automatica (sovrascrive speakers_map.txt)
    auto_path = os.path.join(project_dir, "speakers_map.txt")

    with open(auto_path, 'w', encoding='utf-8') as f:
        f.write("######################################################################\n")
        f.write("# MAPPA SPEAKER AUTOMATICA - GENERATA DA ANALISI EMBEDDING\n")
        f.write("######################################################################\n")
        f.write("# Questa mappatura è generata automaticamente analizzando la similarità\n")
        f.write("# vocale tra i campioni speaker. VERIFICARE MANUALMENTE se necessario.\n")
        f.write("# \n")
        f.write("# CLUSTER TROVATI:\n")

        # Raggruppa per cluster per report
        cluster_speakers = defaultdict(list)
        for block_speaker, cluster_name in suggested_mapping.items():
            cluster_speakers[cluster_name].append(block_speaker)

        for cluster_name, speakers in cluster_speakers.items():
            f.write(f"# {cluster_name}: {len(speakers)} speaker -> {', '.join(speakers)}\n")

        f.write("# \n")
        f.write("######################################################################\n\n")

        # Raggruppa per blocco per formato standard
        blocks_data = defaultdict(dict)
        for block_speaker, suggested_name in suggested_mapping.items():
            block_id = block_speaker.split('.')[0].replace('BLOCK_', '')
            speaker_id = block_speaker.split('.')[1]
            blocks_data[block_id][speaker_id] = suggested_name

        # Scrivi in formato standard
        for block_id in sorted(blocks_data.keys(), key=lambda x: int(x)):
            f.write(f"# BLOCCO {block_id}\n")
            for speaker_id in sorted(blocks_data[block_id].keys()):
                f.write(f"BLOCK_{block_id}.{speaker_id} => {blocks_data[block_id][speaker_id]}\n")
            f.write("\n")

    # Salva anche una copia suggerita
    suggested_path = os.path.join(project_dir, "suggested_speakers_map.txt")
    shutil.copy(auto_path, suggested_path)

    print(f"Mappatura automatica salvata: {auto_path}")
    print(f"Copia suggerita salvata: {suggested_path}")

    return auto_path


def calculate_block_distribution(total_duration, block_duration, residual_threshold=5):
    """
    Calcola la distribuzione ottimale dei blocchi
    Restituisce lista di durate per ogni blocco
    """
    if total_duration <= block_duration:
        return [total_duration]
    
    num_full_blocks = int(total_duration // block_duration)
    residual = total_duration % block_duration
    
    # Se il residuo è significativo (> 1/residual_threshold del blocco), crea blocco aggiuntivo
    if residual > (block_duration / residual_threshold):
        return [block_duration] * num_full_blocks + [residual]
    else:
        # Distribuisci il residuo tra gli ultimi blocchi
        blocks = [block_duration] * num_full_blocks
        if blocks:
            # Aggiungi il residuo all'ultimo blocco
            blocks[-1] += residual
        return blocks


def main():
    parser = argparse.ArgumentParser(
        description="FASE 0 - Diarizzazione a blocchi fissi",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--input", type=str, required=True, help="Percorso del file audio")
    parser.add_argument("--project-dir", "--output-dir", type=str, required=True, help="Directory di progetto")
    
    # Modifica: un solo parametro tra --block-duration e --num-blocks
    block_group = parser.add_mutually_exclusive_group(required=True)
    block_group.add_argument("--block-duration", type=str, 
                           help="Durata target blocchi (es: 1800, 1800s, 30m)")
    block_group.add_argument("--num-blocks", type=int,
                           help="Numero di blocchi in cui dividere l'audio")
    
    parser.add_argument("--sample-duration", type=float, default=60.0,
                       help="Durata campioni audio speaker (secondi) [default: 60]")
    parser.add_argument("--min-speakers", type=int, help="Numero minimo di speaker")
    parser.add_argument("--max-speakers", type=int, help="Numero massimo di speaker")
    parser.add_argument("--num-speakers", type=int, help="Numero esatto di speaker")
    parser.add_argument("--exclusive-mode", action="store_true",
                       help="Usa la diarizzazione exclusive di community-1 per migliore compatibilità Whisper\n"
                            "Unisce il parlato sovrapposto in turni sequenziali")
    parser.add_argument("--quiet", action="store_true", help="Disabilita output dettagliato")
    parser.add_argument("--cpu", action="store_true", help="Forza l'uso della CPU")
    parser.add_argument("--token", type=str, help="Token per Hugging Face Hub")
    parser.add_argument("--force", action="store_true", help="Sovrascrive file esistenti senza chiedere conferma")
    parser.add_argument("--residual-threshold", type=int, default=5,
                       help="Soglia per gestione residuo (default: 5 = 1/5 della durata blocco)")
    parser.add_argument("--auto-map", action="store_true",
                       help="Genera automaticamente la mappatura speaker usando embedding e clustering")
    
    args = parser.parse_args()

    
    # Crea directory progetto
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Gestione configurazione
    config = load_or_create_config(args.project_dir, args)
    
    # Verifica parametri obbligatori
    if not check_required_params(config, args):
        sys.exit(1)
   
    # Directory dei blocchi
    blocks_dir = os.path.join(args.project_dir, "blocks")

    # Verifica se esiste già e chiede conferma per la sovrascrittura
    if not common_utils.check_existing_output(blocks_dir, args.force):
        return 1

    # Considera che l'esecuzione parta qui, dopo la richiesta di conferma
    start_time = time.time()

    # Crea directory blocks e verifica sovrascrittura
    os.makedirs(blocks_dir, exist_ok=True)


    # Ottieni parametri (CLI override config)
    phase0_config = config.get('phase0', {})
    mytoken = get_hf_token(args, config)
    block_duration_str = args.block_duration or phase0_config.get('block_duration')
    num_blocks = args.num_blocks or phase0_config.get('num_blocks')
    sample_duration = args.sample_duration or phase0_config.get('sample_duration', 60.0)
    residual_threshold = args.residual_threshold or phase0_config.get('residual_threshold', 5)
    
    # Converti input a WAV
    input_basename = os.path.basename(args.input)
    converted_audio_path = os.path.join(args.project_dir, "audio_converted.wav")
    
    if not args.quiet:
        print(f"Conversione a WAV: {converted_audio_path}")
    
    cmd = f"ffmpeg -i '{args.input}' -acodec pcm_s16le -ar 16000 -ac 1 -y '{converted_audio_path}' 2>/dev/null"
    subprocess.run(cmd, shell=True, check=True)
    
    # Aggiorna percorso audio convertito nel config
    config['paths']['converted_audio'] = "audio_converted.wav"
    with open(os.path.join(args.project_dir, "config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Ottieni durata totale audio
    audio = AudioSegment.from_wav(converted_audio_path)
    total_duration_sec = len(audio) / 1000.0
    
    # Calcola durata blocco e distribuzione
    if block_duration_str:
        block_duration_sec = common_utils.parse_duration(block_duration_str)
        block_durations = calculate_block_distribution(total_duration_sec, block_duration_sec, residual_threshold)
        num_blocks = len(block_durations)
    else:
        # num_blocks = num_blocks
        block_duration_sec = total_duration_sec / num_blocks
        block_durations = [block_duration_sec] * num_blocks
        # Aggiusta l'ultimo blocco per il residuo
        total_calculated = sum(block_durations)
        if total_calculated < total_duration_sec:
            block_durations[-1] += total_duration_sec - total_calculated
    
    if not args.quiet:
        print(f"Durata audio totale: {total_duration_sec:.1f}s")
        print(f"Numero blocchi: {num_blocks}")
        print(f"Distribuzione blocchi: {[f'{d:.1f}s' for d in block_durations]}")
        print(f"Modalità diarizzazione: {'EXCLUSIVE' if args.exclusive_mode else 'REGULAR'}")
    
    use_gpu = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_gpu else "cpu")
    
    if not args.quiet:
        print(f"Dispositivo: {device}")
    
    
    # Prepara pipeline PyAnnote 4.0 con il nuovo modello community
    print(f"Caricamento modello PyAnnote: pyannote/speaker-diarization-community-1")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=mytoken
        ).to(device)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare modello PyAnnote: {e}")
        return 1
    
    pipeline_params = {}
    if args.num_speakers:
        pipeline_params["num_speakers"] = args.num_speakers
    elif args.min_speakers or args.max_speakers:
        if args.min_speakers:
            pipeline_params["min_speakers"] = args.min_speakers
        if args.max_speakers:
            pipeline_params["max_speakers"] = args.max_speakers
    
    # Processa audio a blocchi
    blocks_data = []
    current_time = 0.0
    current_id_offset = 0
    
    if not args.quiet:
        print("Elaborazione blocchi...\n")
    
    for block_id, actual_duration in enumerate(block_durations):
        if not args.quiet:
            print(f"Blocco {block_id:02d}: da {current_time:.1f}s a circa {current_time+actual_duration:.1f}s (durata: {actual_duration:.1f}s)")

        now = time.time()
        
        # Processa il blocco corrente
        segments, next_start, last_id, speaker_embeddings = process_audio_block(
            converted_audio_path, current_time, actual_duration, total_duration_sec,
            pipeline, device, pipeline_params, current_id_offset, args.exclusive_mode
        )

        gpu_time = time.time() - now

        # Aggiorna offset per il prossimo blocco
        current_id_offset = last_id if last_id > current_id_offset else current_id_offset        
        
        # Crea directory per il blocco
        block_dir = os.path.join(blocks_dir, f"BLOCK_{block_id:02d}")
        os.makedirs(block_dir, exist_ok=True)
        
        # Estrai campioni audio per speaker
        speaker_samples = extract_speaker_samples(
            converted_audio_path, segments, block_dir, sample_duration
        )
        
        # Salva dati del blocco
        block_data = {
            "metadata": {
                "block_id": block_id,
                "start_time": current_time,
                "end_time": next_start,
                "duration": next_start - current_time,
                "num_speakers": len(set(seg['speaker'] for seg in segments)),
                "num_segments": len(segments)
            },
            "segments": segments,
            "speaker_samples": speaker_samples,
            "speaker_embeddings": speaker_embeddings  # <-- AGGIUNTO
        }
        
        json_path = os.path.join(block_dir, f"block_{block_id:02d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(block_data, f, indent=2, ensure_ascii=False)
        
        blocks_data.append(block_data)

        # Genera report statistiche
        report_path = generate_block_report(block_data, block_dir)
        if not args.quiet:
            print(f"  Report: {os.path.basename(report_path)}")
   
        total_time = time.time() - now
        if not args.quiet:
            print(f"  Speaker individuati: {block_data['metadata']['num_speakers']}")
            print(f"  Segmenti estratti: {len(segments)}")
            print(f"  Tempo di GPU: {gpu_time:.1f}s")
            print(f"  Tempo totale: {total_time:.1f}s")
            if block_id < len(block_durations) - 1:
                print(f"  Il prossimo blocco parte da: {next_start:.1f}s\n")
            else:
                print(f"  Questo è l'ultimo blocco (fine blocco a {next_start:.1f}s)\n")
        
        # Prepara per il prossimo blocco
        current_time = next_start

    
    # Genera mappa speaker: automatica o manuale?
    if args.auto_map:
        # Controlla se c'è più di un blocco
        if len(blocks_data) <= 1:
            print("Auto-mapping degli speaker disattivato con un solo blocco")
            map_path = generate_speaker_map(blocks_data, args.project_dir)
        else:
            print("Generazione mappatura automatica speaker...")
            # Estrai embedding e genera mappatura automatica
            #embeddings_dict, sample_paths = extract_embeddings_from_samples(
            #    args.project_dir, mytoken, blocks_data
            #)            
            embeddings_dict = extract_embeddings_from_diarization(blocks_data)
            if embeddings_dict:
                suggested_mapping, cluster_groups = generate_auto_mapping(embeddings_dict)
                map_path = save_auto_mapping(args.project_dir, suggested_mapping, cluster_groups)
                print(f"Mappatura automatica completata: {len(suggested_mapping)} speaker mappati")
            else:
                print("Fallback alla mappatura manuale per errore nell'estrazione embedding.")
                map_path = generate_speaker_map(blocks_data, args.project_dir)
    else:
        map_path = generate_speaker_map(blocks_data, args.project_dir)

    # Salva metadata complessivo

    actual_block_durations = []
    for block_data in blocks_data:
        start = block_data['metadata']['start_time']
        end = block_data['metadata']['end_time']
        actual_duration = end - start
        actual_block_durations.append(actual_duration)

    metadata = {
        "source_file": args.input,
        "wav_file": converted_audio_path,
        "total_duration": total_duration_sec,
        "block_duration": block_duration_sec,
        "num_blocks": len(blocks_data),
        "total_segments": sum(len(block['segments']) for block in blocks_data),
        "processing_time": time.perf_counter() - start_time,
        "exclusive_mode": args.exclusive_mode,
        "residual_threshold": residual_threshold,
        "actual_block_durations": actual_block_durations
    }
    
    metadata_path = os.path.join(args.project_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Tempo totale
    elapsed_time = common_utils.save_execution_stats(args.project_dir, os.path.basename(sys.argv[0]), start_time, args)
    
    if not args.quiet:
        print(f"\n=== FASE 0 COMPLETATA ===")
        print(f"Blocchi processati: {len(blocks_data)}")
        print(f"Segmenti totali: {metadata['total_segments']}")
        print(f"Modalità: {'EXCLUSIVE' if args.exclusive_mode else 'REGULAR'}")
        print(f"Mappa speaker: {map_path}")
        print(f"Tempo totale: {elapsed_time:.2f}s")
    
    return 0


if __name__ == "__main__":
    main()
