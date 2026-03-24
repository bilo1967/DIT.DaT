import argparse
import fase0
import fase1
import fase2
import fase3
import fase4
import fase5
import fase6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FASE 0 - Diarizzazione a blocchi fissi",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--input", type=str, required=True, help="Percorso del file audio")
    parser.add_argument("--project-dir", type=str, required=True, help="Directory di progetto")

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
    parser.add_argument("--speaker-map", help="File speakers_map.txt (opzionale, default: project-dir/speakers_map.txt)")
    parser.add_argument("--output-json", help="File JSON di output (opzionale, default: project-dir/fase1_unified.json)")
    parser.add_argument("--input-json", help="Input JSON dalla Fase 1 (opzionale, default: project-dir/fase1_unified.json)")
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
    parser.add_argument("--input-wav", help="File audio WAV (opzionale, default: dal config o JSON)")
    parser.add_argument("--output-dir", help="Directory di output (opzionale, default: project-dir/combined)")
    parser.add_argument("--fill-mode", choices=["none", "white", "pink"], default="none",
                       help="Modalità di riempimento silenzi: none=silenzio, white=rumore bianco, pink=rumore rosa")
    parser.add_argument("--speaker", help="Processa solo questo speaker (opzionale)")
    parser.add_argument("--dump-segments", action="store_true",
                       help="Salva i singoli frammenti audio (default: solo composito)")

    # Input/Output
    parser.add_argument("--input-json-fase2", help="JSON file dalla Fase 2 (opzionale, default: project-dir/fase2_filtered.json)")

    # Parametri Whisper
    parser.add_argument("--model", help="Modello Whisper (default: dal config o large-v3)")
    parser.add_argument("--language", help="Lingua audio (default: dal config o de)")
    parser.add_argument("--temperature", type=float, help="Temperature Whisper")
    parser.add_argument("--beam_size", type=int, help="Beam size Whisper")
    parser.add_argument("--best_of", type=int, help="Best of Whisper")
    parser.add_argument("--no_speech_threshold", type=float, help="No speech threshold")
    parser.add_argument("--compression_ratio_threshold", type=float, help="Compression ratio threshold")
    parser.add_argument("--word_timestamps", action="store_true", help="Includi timestamp a livello di parola")
    parser.add_argument("--input-dir",
        help="Directory input trascrizioni (opzionale, default: project-dir/transcripts)")
    parser.add_argument("--use-whisper-segments", action="store_true",
        help="Usa i segmenti Whisper dettagliati invece di quelli uniti")

    parser.add_argument("--input6",
           help="File SRT revisionato (default: project-dir/subs/podcast_complete.srt)")
    args = parser.parse_args()

    fase0.main(args)
    fase1.main(args)
    fase2.main(args)
    fase3.main(args)
    # fase4.main(args)
    # fase5.main(args)
    # fase6.main(args)
