1. create a virtual environment with python>=3.12
   `python3.12 -m venv .venv`
2. activate the virtual environment
   `source .venv/bin/activate` on unix-like OS
3. install the package
   `pip install -e .`
4. run `ditdat`
   ```
   ditdat -h
	usage: ditdat [-h] [--version] COMMAND ...

	DIT.DaT — Speaker diarization and transcription pipeline

	positional arguments:
	COMMAND
		phase0         Diarization and block splitting
		phase1         Unify blocks and apply speaker mapping
		phase2         Merge segments and filter short ones
		phase2-report  Generate interactive HTML report
		phase3         Extract per-speaker audio files
		phase4         Whisper transcription
		phase5         Generate SRT and TXT output files
		phase6         Validate and regenerate subtitles after revision
		diarize        Split per-speaker audio (phase0 through phase3)

	options:
	-h, --help       show this help message and exit
	--version        show program's version number and exit
  	```