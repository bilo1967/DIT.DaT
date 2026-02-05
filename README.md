<<<<<<< HEAD
# DIT.DaT tool

**DIT Diarization and Transcription tool**

DIT.DaT is an experimental, modular pipeline for **speaker diarization and automatic transcription of long-form audio**, designed to support students and researchers working on spoken-language data.

The project was developed to process **very long recordings** (e.g. multi-hour interviews and podcasts), preserving speaker separation, producing revision-friendly outputs, and minimizing—but not eliminating—the need for manual intervention.

---

## Project background

DIT.DaT was developed by **Gabriele Carioli** for the **Department of Interpretation and Translation (DIT)** of the **University of Bologna**.

The original use case involved academic podcasts and interviews lasting three hours or more, where commercial cloud-based ASR services were either too costly, opaque, or unsuitable for long-term academic workflows. The project therefore focuses exclusively on **open-source AI models** that can be executed on local hardware.

---

## Key features

* Speaker diarization for long audio files (hours-long)
* Robust handling of diarization drift via block-based processing
* Semi-automatic and experimental automatic speaker mapping across blocks
* Two-stage segment cleanup to reduce fragmentation
* High-quality multilingual transcription using Whisper
* Outputs optimized for **manual revision** using subtitle editors
* Fully persistent, restartable, and inspectable processing pipeline

---

## Technology stack

DIT.DaT combines two mature open-source systems with complementary strengths:

### PyAnnote Audio

* Used for **speaker diarization** ("who speaks when")
* Model: `speaker-diarization-community-1`
* Supports **exclusive mode** to handle overlapping speech
* Produces speaker embeddings reused for cross-block speaker mapping

### Whisper (OpenAI)

* Used for **automatic speech recognition (ASR)**
* Model: `large-v3`
* Multilingual, end-to-end encoder–decoder architecture
* Particularly effective on long-form and academic speech

---

## Hardware requirements

* **GPU strongly recommended**
* The pipeline is designed to run on dedicated workstations
* CPU-only execution is technically possible but impractical for long audio

No specific GPU brand is enforced, but adequate VRAM is required for large models.

---

## Design principles

* **Modular pipeline**, not a monolithic application
* Explicit handling of known model limitations
* Human-in-the-loop where necessary and meaningful
* Full transparency via intermediate artifacts
* Reproducibility through versioned configuration

The system is intentionally designed so that every phase can be inspected, corrected, or re-run independently.

---

## Pipeline overview

Processing is organized into a sequence of independent phases, all operating inside a dedicated **project directory**.

Each phase reads the output of the previous one and produces persistent artifacts.

### Phase 0 – Preparation and diarization

* Audio conversion to WAV (16 kHz, mono)
* Optional splitting into fixed-duration blocks (default: 30 minutes)
* Independent diarization of each block
* Extraction of speaker audio samples for manual identification

### Phase 1 – Unification

* Applies the speaker mapping across blocks
* Preserves original segment identifiers for traceability

### Phase 2 – Cleanup

* Merges consecutive segments from the same speaker
* Removes very short residual segments
* Allows manual removal or merging of speakers

### Phase 3 – Combined audio extraction (optional)

* Generates one audio track per speaker
* Useful for analysis or downstream processing

### Phase 4 – Transcription

* Speaker-specific segments passed to Whisper
* Padding added to improve recognition on short segments
* Produces both raw and aggregated transcripts

### Phase 5 – Export

* Generates SRT subtitles and text files
* Outputs compatible with SubtitleEdit and similar tools

---

## Speaker mapping

When block-based processing is enabled, speakers are identified independently in each block.

DIT.DaT therefore provides:

* **Manual speaker mapping** via editable text files
* **Experimental automatic mapping** based on speaker embeddings

Automatic mapping can significantly reduce manual work but still requires verification.

---

## Output formats

Final outputs include:

* JSON files with full transcripts and metadata
* SRT subtitle files for revision
* Text files with and without timestamps
* Clean text files suitable for publication
* Optional HTML reports for diarization analysis

---

## Known limitations

Limitations are largely intrinsic to the underlying models:

* Output quality depends strongly on audio quality
* Reduced accuracy with strong regional accents
* Difficulty distinguishing very similar voices
* Imperfect handling of music and sound effects

As a result, **manual revision is always expected**.

DIT.DaT aims to minimize, not eliminate, human intervention.

---

## License

This project is released under the **MIT License**.

---

## Project status

**Experimental / research prototype**

The tool is actively usable but not designed as a turnkey consumer application. It is intended for technically competent users in academic or research contexts.

=======
# DIT.DaT
DIT - Diarization and Transcription Tool: a pipeline of scripts for diarizing and transcribing a long audio file, using Whisper and PyAnnote.
>>>>>>> e7429cbc39d1a6cf3a91bf779ec63f87fd7ca594
