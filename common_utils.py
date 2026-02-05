#common_utils.py

#
# Funzioni comuni a tutte le fasi
#

import json
import os
import time
import sys
import pprint
import numpy as np
from datetime import datetime
from argparse import Namespace
from pydub.generators import WhiteNoise
from pydub import AudioSegment
from typing import Any




# Esempio di utilizzo:
# dump_object(diarization)
def dump_object(obj: Any, max_items: int = 3) -> None:
    """
    Analizza e stampa la struttura di un oggetto in modo dettagliato.

    Args:
        obj: L'oggetto da analizzare
        max_items: Numero massimo di elementi da mostrare per iterabili
    """
    print("=" * 60)
    print("ANALISI OGGETTO")
    print("=" * 60)

    # 1. Tipo dell'oggetto
    print(f"1. TIPO: {type(obj)}")
    print()

    # 2. Attributi e metodi disponibili
    print("2. ATTRIBUTI E METODI:")
    attributes = dir(obj)
    pprint.pprint(attributes)
    print()

    # 3. Esplora gli attributi non privati con valori
    print("3. ATTRIBUTI NON PRIVATI CON VALORI:")
    for attr in attributes:
        if not attr.startswith('_'):
            try:
                value = getattr(obj, attr)
                if not callable(value):  # Escludi i metodi
                    print(f"   {attr}: {type(value)} = {value}")
            except Exception as e:
                print(f"   {attr}: <ERRORE - {e}>")
    print()

    # 4. Verifica se è iterabile
    print("4. ANALISI ITERABILE:")
    try:
        items = list(obj)
        print(f"   È iterabile con {len(items)} elementi")
        print(f"   Primi {min(max_items, len(items))} elementi:")
        for i, item in enumerate(items[:max_items]):
            print(f"     [{i}]: {type(item)} = {item}")
        if len(items) > max_items:
            print(f"     ... e altri {len(items) - max_items} elementi")
    except TypeError:
        print("   Non è iterabile")
    except Exception as e:
        print(f"   Errore durante l'iterazione: {e}")
    print()

    # 5. Tentativo di serializzazione JSON
    print("5. TENTATIVO SERIALIZZAZIONE JSON:")
    try:
        # Prova vari approcci di serializzazione
        if hasattr(obj, '__dict__'):
            data = obj.__dict__
        elif hasattr(obj, '_asdict'):  # Per namedtuple
            data = obj._asdict()
        else:
            data = dict(obj) if hasattr(obj, '__iter__') and not isinstance(obj, str) else str(obj)

        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        print("   Successo:")
        print(json_str)
    except Exception as e:
        print(f"   Fallito: {e}")
        # Ultima risorsa: rappresentazione stringa
        try:
            print(f"   Rappresentazione stringa: {str(obj)}")
        except:
            print("   Impossibile ottenere rappresentazione stringa")
    print()

    # 6. Metodi speciali
    print("6. METODI SPECIALI:")
    special_methods = [attr for attr in attributes if attr.startswith('__') and attr.endswith('__')]
    print(f"   Trovati {len(special_methods)} metodi speciali")
    print()





def parse_duration(duration_str):
    """Parsa durate in secondi da stringhe come '1800', '1800s', '30m'"""
    duration_str = duration_str.lower().strip()
    
    if duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith('s'):
        return float(duration_str[:-1])
    else:
        return float(duration_str)



def calculate_checksum(file_path, bytes_to_read=8192):
    """Calcola checksum parziale (primi X bytes)"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        hash_md5.update(f.read(bytes_to_read))
    return hash_md5.hexdigest()[:8]






def save_execution_stats(project_dir: str, script_name: str, start_time: float, args: Namespace):
    """
    Salva le statistiche di esecuzione nel file stats.json
    Aggiorna solo la chiave dello script specifico, mantiene le altre
    """
    stats_file = os.path.join(project_dir, "stats.json")
    
    # Calcola durata
    end_time = time.time()
    duration = end_time - start_time
    
    # Prepara i parametri (escludendo project_dir)
    parameters = {}
    for key, value in vars(args).items():
        if key != 'project_dir':
            # Converti valori non serializzabili in stringa
            try:
                json.dumps(value)
                parameters[key] = value
            except (TypeError, ValueError):
                parameters[key] = str(value)
    
    # Crea oggetto statistiche per questo script
    script_stats = {
        "timestamp": datetime.now().isoformat(),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": round(duration, 2),
        "parameters": parameters
    }
    
    # Carica stats esistenti o crea nuovo dict
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
        except (json.JSONDecodeError, Exception):
            stats_data = {}
    else:
        stats_data = {}
    
    # Aggiorna solo la chiave di questo script
    stats_data[script_name] = script_stats
    
    # Salva il file aggiornato
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    return duration

# === FUNZIONI FILE SYSTEM ===
def check_existing_output(output_path, force_overwrite):
    """Verifica se il file/directory di output esiste già"""
    if os.path.exists(output_path):
        if os.path.isfile(output_path):
            # È un file
            if not force_overwrite:
                user_input = input(f"File {output_path} esiste già. Sovrascrivere? [y/N] ")
                if user_input.lower() != 'y' and user_input.lower() != 's':
                    print("Operazione annullata")
                    return False
            return True
        else:
            # È una directory
            if any(os.listdir(output_path)) and not force_overwrite:
                user_input = input(f"Directory {output_path} non vuota. Sovrascrivere? [y/N] ")
                if user_input.lower() != 'y' and user_input.lower() != 's':
                    print("Operazione annullata")
                    return False
            return True
    return True


def load_config(project_dir):
    """Carica il file di configurazione del progetto"""
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"ERRORE: File config.yaml non trovato in {project_dir}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            import yaml
            return yaml.safe_load(f)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare config.yaml: {e}")
        return None

def get_input_json_path(project_dir, args, default_name="fase2_filtered.json"):
    """Ottiene il percorso del file JSON di input"""
    if hasattr(args, 'input_json') and args.input_json:
        return args.input_json
    
    # Cerca il file predefinito
    default_path = os.path.join(project_dir, default_name)
    if os.path.exists(default_path):
        return default_path
    
    return None




def ensure_numpy_array(embedding_data):
    """Converte i dati embedding in array numpy, gestendo diversi formati"""
    if isinstance(embedding_data, np.ndarray):
        return embedding_data
    elif isinstance(embedding_data, (list, tuple)):
        return np.array(embedding_data)
    else:
        # Prova a convertire in qualsiasi altro caso
        return np.array(embedding_data)




# === FUNZIONI AUDIO ===
def generate_pink_noise(duration_ms, volume_db=-50):
    """Genera rumore rosa filtrando rumore bianco"""
    noise = WhiteNoise().to_audio_segment(duration=duration_ms)
    return noise.low_pass_filter(1000).apply_gain(volume_db)

def generate_silence(duration_ms, fill_mode):
    """Crea segmento di silenzio/rumore in base alla modalità"""
    if fill_mode == "none":
        return AudioSegment.silent(duration=duration_ms)
    elif fill_mode == "white":
        return WhiteNoise().to_audio_segment(duration=duration_ms).apply_gain(-40)
    elif fill_mode == "pink":
        return generate_pink_noise(duration_ms)
    else:
        raise ValueError(f"Modalità fill non valida: {fill_mode}")

def export_audio_segment(audio, output_path, format="wav"):
    """Esporta segmento audio nel formato specificato"""
    audio.export(output_path, format=format, parameters=["-ar", "16000", "-ac", "1"])

