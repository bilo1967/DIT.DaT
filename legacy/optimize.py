from argparse import Namespace
import preprocess
import align
import argparse

import fase4
import fase5
import fase6

import optuna
import json
import numpy as np
import os
import uuid



def sample_params(trial):
	return {
		"temperature": trial.suggest_float("temperature", 0.0, 1.0),
		"beam_size": trial.suggest_int("beam_size", 1, 10),
		"best_of": trial.suggest_int("best_of", 1, 10),
		"no_speech_threshold": trial.suggest_float("no_speech_threshold", 0.2, 0.8),
		"compression_ratio_threshold": trial.suggest_float("compression_ratio_threshold", 1.5, 2.8),
		"model": trial.suggest_categorical("model", ["small", "medium", "large-v3", "turbo"]),
		"condition_on_previous_text": trial.suggest_categorical("condition_on_previous_text", [True, False]),
		"patience": trial.suggest_float("patience", 0.2, 0.8),
		"length_penalty": trial.suggest_float("length_penalty", 0.2, 0.8)
	}

def wrapper_fasi(file_da_trascrivere, args, temperature, beam_size, best_of, no_speech_threshold, compression_ratio_threshold, model, condition_on_previous_text, patience, length_penalty):

	new_args = Namespace(
		temperature=temperature,
		beam_size=beam_size,
		best_of=best_of,
		no_speech_threshold=no_speech_threshold,
		compression_ratio_threshold=compression_ratio_threshold,
		force=args.force,
		language=args.language,
		word_timestamps=args.word_timestamps,
		model=model,
		speaker=args.speaker,
		input_json=args.input_json,
		input_dir=args.input_dir,
		project_dir=args.project_dir,
		input_wav = args.input_wav,
		output_dir=args.output_dir,
		use_whisper_segments=args.use_whisper_segments,
		condition_on_previous_text = condition_on_previous_text,
		patience = patience,
		length_penalty = length_penalty,
		suffix = args.suffix
		)

	# print("ARGS:", new_args)
	# input()

	fase4.main(new_args)
	new_args.output_dir = f"{new_args.project_dir}/subs_{new_args.suffix}"
	fase5.main(new_args)
	new_args.input6 = f"{new_args.project_dir}/subs_{new_args.suffix}/podcast_complete.srt"
	new_args.output_dir = f"{new_args.project_dir}/subs_final_{new_args.suffix}"
	fase6.main(new_args)


def compute_score(trascrizione_gold_path, trascrizione_whi_path):

	normalized_whisper = preprocess.main(trascrizione_whi_path)

	with open(trascrizione_gold_path, encoding='utf-8') as fin:
		content_gold = fin.readlines()

	content_whisper = [x.strip().split('\t', 1) for x in normalized_whisper]
	content_gold = [x.strip().split('\t', 1) for x in content_gold]
	content_gold = [x for x in content_gold if not '?' in x[0]]

	speaker_gold = []
	for speaker, line in content_gold:
		if not speaker in speaker_gold:
			speaker_gold.append(speaker)

	speaker_whisper = []
	for speaker, line in content_whisper:
		if not speaker in speaker_whisper:
			speaker_whisper.append(speaker)

	speaker_map = dict(zip(speaker_gold, speaker_whisper))

	all_words_whisper = []
	all_words_gold = []

	for speaker, line in content_gold:
		linesplit = line.split()
		for w in linesplit:
			all_words_gold.append((speaker, w))

	for speaker, line in content_whisper:
		linesplit = line.split()
		for w in linesplit:
			all_words_whisper.append((speaker, w))

	aligned_whisper, aligned_gold = align.align_words(all_words_whisper, all_words_gold)


	wer = align.compute_wer(aligned_whisper, aligned_gold)

	# MARTINA: penso che devi adattare la funzione align.process_file o qualcosa del genere,
	# aggiungendo un suffisso al nome del file in modo che ogni trial abbia il suo nome unico, tipo
	# aligned_BOC1002_temp551_beam2_nospeech47 etc...
	# align.print_alignment(aligned_whisper, aligned_gold)
	# su questo file poi puoi calcolare BLEU, altre misur di accuracy etc...

	return wer

def evaluate_dataset_with_pruning(trial, audio_paths, references, params, args, chunk_size=7):

	wers = []
	for i in range(0, len(audio_paths), chunk_size):
		chunk_paths = audio_paths[i:i+chunk_size]
		chunk_refs  = references[i:i+chunk_size]

		# Evaluate this chunk
		for ap, ref in zip(chunk_paths, chunk_refs):
			# SET ARGS:
			args.project_dir = f"{ap}_minduration5_minpause2" #! CHANGE HERE IF CHANGES ARE MADE FOR STEPS 1-3
			args.input = f"audio20min/{ap}.mp3"
			args.output_dir = f"{args.project_dir}/transcripts_{trial.number}"
			args.suffix = trial.number

			# AUDIO FILE
			wrapper_fasi(ap, args, **params)
			# print("ARGS:", args)
			# input()
			wers.append(compute_score(ref,
                             f"{args.project_dir}/subs_final_{trial.number}/podcast_clean_validated.txt"))

		# Intermediate objective (mean WER so far)
		interim = float(np.mean(wers))

		# Report progress to Optuna
		step = (i // chunk_size) + 1
		trial.report(interim, step=step)

		# Ask Optuna whether to prune
		if trial.should_prune():
			raise optuna.TrialPruned()

	return float(np.mean(wers))


def main(args):

	# # PROVA BOC1002
	# manual_params = {
	#     "temperature": 0.5512052052560472,
	#     "beam_size": 2,
	#     "best_of": 4,
	#     "no_speech_threshold": 0.47468096277550004,
	#     "compression_ratio_threshold": 2.591515778575058
	# }

	# print("Running MANUAL decoding test with parameters:")
	# for k, v in manual_params.items():
	#     print(f"  {k}: {v}")

	# wrapper_fasi(
	#     args.input,
	#     args,
	#     **manual_params
	# )

	# return

	val_audio = ["BOC1002"]
	# val_audio = ["BOC1002", "BOC1003", "BOC1005"]
	# BOC1002/subs_final/podcast_clean_validated.txt
	val_text = [f"GOLD/{conv_id}.txt" for conv_id in val_audio]

	def objective(trial):
		params = sample_params(trial)
		trial_id = trial.number
		return evaluate_dataset_with_pruning(trial, val_audio, val_text, params, args)


	study = optuna.create_study(
		direction="minimize",
		sampler=optuna.samplers.TPESampler(),
		pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
	)

	study.optimize(objective, n_trials=50)

	trials = [
		{
			"trial": t.number,
			"value": t.value,
			"params": t.params,
			"state": t.state.name,
		}
    	for t in study.trials
	]

	with open(os.path.join(args.project_dir, "optuna_trials.json"), "w") as f:
		json.dump(trials, f, indent=2)

	with open(os.path.join(args.project_dir, "parameters.json"), 'w') as fout:
		json.dump(study.best_params, fout, indent=2)


if __name__ == '__main__':
#	fpath_whisper = "/home/msimonotti/progetto/BOC1002/subs_final/podcast_clean_validated.txt"
#	fpath_gold = "/home/msimonotti/progetto/GOLD/BOC1002.txt"
#	compute_score(fpath_gold, fpath_whisper)

	parser = argparse.ArgumentParser(
		description="FASE 0 - Diarizzazione a blocchi fissi",
		formatter_class=argparse.RawTextHelpFormatter
	)

	parser.add_argument("--input", type=str, help="Percorso del file audio")
	parser.add_argument("--project-dir", type=str, help="Directory di progetto")

	# Modifica: un solo parametro tra --block-duration e --num-blocks
	block_group = parser.add_mutually_exclusive_group()
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

	main(args)
