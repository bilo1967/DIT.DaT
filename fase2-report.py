#!/usr/bin/env python3
"""
FASE 2 REPORT - Genera report HTML interattivo con waveform e timeline
Visualizza segmenti degli speaker da fase1 e fase2 per analisi comparativa
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
import yaml

# Funzioni comuni
import common_utils

def get_wav_file_path(project_dir):
    """Ottiene il percorso del file WAV dal config"""
    config = common_utils.load_config(project_dir)
    if config and 'paths' in config and 'converted_audio' in config['paths']:
        wav_path = os.path.join(project_dir, config['paths']['converted_audio'])
        if os.path.exists(wav_path):
            return wav_path
    
    # Fallback: cerca qualsiasi file WAV nella project directory
    for wav_file in Path(project_dir).glob("*.wav"):
        return str(wav_file)
    return None

def load_json_data(json_path):
    """Carica dati JSON mantenendo la formattazione originale"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sample_audio_waveform(wav_path, sample_rate=10):
    """
    Campiona l'audio per generare la waveform
    sample_rate: Hz (campioni al secondo) - default 10Hz = 0.1s per pixel
    """
    try:
        from pydub import AudioSegment
        import numpy as np
    except ImportError:
        print("ERRORE: pydub non installato. Installa con: pip install pydub")
        return None

    # Carica audio
    audio = AudioSegment.from_wav(wav_path)
    duration_sec = len(audio) / 1000.0
    
    # Converti in numpy array
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)  # Mono mix
    
    # Calcola numero di campioni per la waveform
    total_pixels = int(duration_sec * sample_rate)
    
    # Riduci risoluzione per la visualizzazione
    original_samples_per_pixel = len(samples) / total_pixels
    
    waveform_data = []
    for i in range(total_pixels):
        start_idx = int(i * original_samples_per_pixel)
        end_idx = int((i + 1) * original_samples_per_pixel)
        segment = samples[start_idx:end_idx]
        
        if len(segment) > 0:
            # Normalizza tra 0 e 1 per la visualizzazione
            max_val = np.max(np.abs(segment))
            if max_val > 0:
                normalized = max_val / 32768.0
            else:
                normalized = 0
            waveform_data.append(float(normalized))
        else:
            waveform_data.append(0.0)
    
    return {
        'data': waveform_data,
        'duration': duration_sec,
        'sample_rate': sample_rate,
        'total_pixels': total_pixels
    }

def get_fallback_template():
    return '''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{FILE_NAME}} - Report post filtraggio</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        .timeline-container {
            width: 100%;
            overflow-x: auto;
            background: white;
        }
        
        .timeline-scroll-content {
            position: relative;
            min-width: 100%;
        }
        
        .waveform-container {
            height: 180px;
            border-bottom: 2px solid #dee2e6;
            position: relative;
            background: #f8f9fa;
            display: flex;
        }
        
        .waveform-segment {
            height: 180px;
            flex-shrink: 0;
        }
        
        .speaker-timeline {
            height: 60px;
            border-bottom: 1px solid #e9ecef;
            position: relative;
            background: white;
        }
        
        .time-markers {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 100;
        }
        
        .time-marker {
            position: absolute;
            top: 0;
            width: 1px;
            height: 100%;
        }
        
        .time-marker.minor {
            background: rgba(0,0,0,0.3);
        }
        
        .time-marker.major {
            background: rgba(0,0,0,0.7);
        }
        
        .time-label {
            position: absolute;
            top: 5px;
            transform: translateX(-50%);
            font-size: 12px;
            color: #333;
            background: rgba(255,255,255,0.95);
            padding: 2px 6px;
            border-radius: 3px;
            white-space: nowrap;
            font-weight: bold;
            border: 1px solid #dee2e6;
        }
        
        .segment {
            position: absolute;
            height: 50px;
            border-radius: 4px;
            opacity: 0.8;
            cursor: pointer;
            transition: opacity 0.2s;
            margin-top: 5px;
        }
        
        .segment:hover {
            opacity: 1;
            z-index: 50;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        .segment-phase1 {
            border: 2px dashed rgba(0,0,0,0.4);
        }
        
        .segment-phase2 {
            border: 2px solid rgba(0,0,0,0.6);
        }
        
        .segment-info {
            position: absolute;
            background: white;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1001;
            font-size: 13px;
            max-width: 350px;
            pointer-events: none;
        }
        
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            background: #f8f9fa;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 25px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }
        
        .controls {
        }
        
        .stats-panel {
            margin-top: 25px;
        }
        
        .debug-info {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.375rem;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
        }
        
        .stats-readable {
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.4;
        }
    </style>

    <script>
        // Dati inseriti dallo script Python
        const waveformData = {{WAVEFORM_DATA}};
        const phase1Data = {{PHASE1_DATA}};
        const phase2Data = {{PHASE2_DATA}};
        const sampleRate = {{SAMPLE_RATE}};
        const fileName = "{{FILE_NAME}}";
        const colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
        ];

        let zoomLevel = 1.0;

        function initTimeline() {
            console.log('Initializing timeline...');
            
            const duration = waveformData.duration;
            const totalPixels = waveformData.total_pixels;
            const scrollContent = $('#scrollContent');
            
            // Imposta larghezza del contenuto scorrevole
            scrollContent.css('width', totalPixels + 'px');
            
            // Disegna i componenti nell'ordine corretto
            drawWaveform();
            drawSpeakerTimelines();
            drawTimeMarkers(duration, totalPixels);
            
            initLegend();
            initControls();
            showStatistics();
            
            // Aggiorna debug info e sposta in fondo
            updateDebugInfo();
        }

        function drawWaveform() {
            const container = $('#waveformContainer');
            container.empty();
            
            const totalPixels = waveformData.total_pixels;
            const data = waveformData.data;
            const height = 180;
            
            console.log('Drawing waveform, total pixels:', totalPixels);
            console.log('Waveform data length:', data.length);
            
            // Dimensione massima del canvas (conservativa per tutti i browser)
            const maxCanvasWidth = 16384; // 16K pixels - sicuro per tutti i browser
            
            if (totalPixels <= maxCanvasWidth) {
                // Caso semplice: un solo canvas
                drawWaveformSegment(container, 0, totalPixels, data, height);
            } else {
                // Dividi in più canvas
                const numSegments = Math.ceil(totalPixels / maxCanvasWidth);
                console.log(`Dividing waveform into ${numSegments} segments`);
                
                for (let i = 0; i < numSegments; i++) {
                    const start = i * maxCanvasWidth;
                    const end = Math.min((i + 1) * maxCanvasWidth, totalPixels);
                    const segmentData = data.slice(start, end);
                    
                    drawWaveformSegment(container, start, end - start, segmentData, height);
                }
            }
        }

        function drawWaveformSegment(container, startX, width, data, height) {
            const canvas = $('<canvas>').addClass('waveform-segment').attr({
                width: width,
                height: height
            });
            container.append(canvas);
            
            const canvasElement = canvas.get(0);
            const ctx = canvasElement.getContext('2d');
            
            console.log(`Drawing segment: ${startX} to ${startX + width}, ${data.length} points`);
            
            // Sfondo
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, width, height);
            
            if (data.length === 0) {
                ctx.fillStyle = '#999';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('No data', width / 2, height / 2);
                return;
            }
            
            // Disegna le linee verticali per ogni campione
            ctx.strokeStyle = '#1e40af';
            ctx.lineWidth = 1;
            
            for (let i = 0; i < data.length; i++) {
                // Clamp del valore tra 0 e 1 e calcola altezza
                const clampedValue = Math.min(data[i], 1.0);
                const lineHeight = clampedValue * height;
                
                // Disegna una linea verticale dal centro
                const yStart = height/2 - lineHeight/2;
                const yEnd = height/2 + lineHeight/2;
                
                ctx.beginPath();
                ctx.moveTo(i, yStart);
                ctx.lineTo(i, yEnd);
                ctx.stroke();
            }
            
            // Linea centrale per riferimento
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 3]);
            ctx.beginPath();
            ctx.moveTo(0, height/2);
            ctx.lineTo(width, height/2);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        function drawSpeakerTimelines() {
            const timelinesContainer = $('#speakerTimelines');
            timelinesContainer.empty();
            
            // Raccogli tutti gli speaker unici
            const allSpeakers = new Set();
            phase1Data.segments.forEach(seg => allSpeakers.add(seg.speaker));
            Object.keys(phase2Data.speakers || {}).forEach(speaker => allSpeakers.add(speaker));
            
            console.log('Speakers found:', Array.from(allSpeakers));
            
            // Crea una timeline per ogni speaker
            Array.from(allSpeakers).forEach((speaker, index) => {
                const timeline = $('<div>').addClass('speaker-timeline')
                    .attr('id', 'timeline-' + speaker.replace(/\s+/g, '-'))
                    .attr('title', 'Speaker: ' + speaker);
                
                // Etichetta speaker a sinistra
                const label = $('<div>').css({
                    position: 'absolute',
                    left: '10px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    fontWeight: 'bold',
                    fontSize: '14px',
                    color: '#333',
                    zIndex: 10
                }).text(speaker);
                timeline.append(label);
                
                // Segmenti Fase 1
                phase1Data.segments
                    .filter(seg => seg.speaker === speaker)
                    .forEach(seg => {
                        drawSegment(timeline, seg, 'phase1', index);
                    });
                
                // Segmenti Fase 2
                if (phase2Data.speakers && phase2Data.speakers[speaker]) {
                    phase2Data.speakers[speaker].segments
                        .forEach(seg => {
                            drawSegment(timeline, seg, 'phase2', index);
                        });
                }
                
                timelinesContainer.append(timeline);
            });
        }

        function drawTimeMarkers(duration, totalPixels) {
            const markersContainer = $('#timeMarkers');
            markersContainer.empty();
            
            const pixelsPerSecond = totalPixels / duration;
            
            console.log('Drawing time markers:', duration, 'seconds,', pixelsPerSecond.toFixed(2), 'px/s');
            
            // Marcatori ogni 30 secondi e ogni minuto
            for (let time = 0; time <= duration; time += 30) {
                const isMajor = time % 60 === 0;
                const left = time * pixelsPerSecond;
                
                const marker = $('<div>').addClass('time-marker')
                    .addClass(isMajor ? 'major' : 'minor')
                    .css('left', left + 'px');
                
                // Mostra etichetta per ogni marcatore (ogni 30 secondi)
                const label = $('<div>').addClass('time-label')
                    .text(formatTime(time))
                    .css('left', left + 'px');
                markersContainer.append(label);
                
                markersContainer.append(marker);
            }
            
            // Aggiungi marcatore finale se necessario
            const finalLeft = duration * pixelsPerSecond;
            const finalMarker = $('<div>').addClass('time-marker major')
                .css('left', finalLeft + 'px');
            const finalLabel = $('<div>').addClass('time-label')
                .text(formatTime(duration))
                .css('left', finalLeft + 'px');
            
            markersContainer.append(finalLabel);
            markersContainer.append(finalMarker);
        }

        function drawSegment(timeline, segment, phase, colorIndex) {
            const duration = waveformData.duration;
            const totalPixels = waveformData.total_pixels;
            const pixelsPerSecond = totalPixels / duration;
            
            const left = segment.start * pixelsPerSecond;
            const width = Math.max((segment.end - segment.start) * pixelsPerSecond, 2);
            
            const segmentDiv = $('<div>').addClass('segment')
                .addClass('segment-' + phase)
                .css({
                    left: (left + 50) + 'px',
                    width: width + 'px',
                    'background-color': colors[colorIndex % colors.length]
                })
                .attr('data-speaker', segment.speaker)
                .attr('data-phase', phase);
            
            // Tooltip
            segmentDiv.hover(
                function(e) {
                    const info = $('<div>').addClass('segment-info')
                        .html(`
                            <strong>${segment.speaker}</strong><br>
                            Fase: ${phase === 'phase1' ? '1 (Originale)' : '2 (Filtrato)'}<br>
                            Inizio: ${segment.start.toFixed(2)}s<br>
                            Fine: ${segment.end.toFixed(2)}s<br>
                            Durata: ${segment.duration.toFixed(2)}s
                        `)
                        .css({
                            left: e.pageX + 10,
                            top: e.pageY + 10
                        });
                    $('body').append(info);
                },
                function() {
                    $('.segment-info').remove();
                }
            );
            
            timeline.append(segmentDiv);
        }

        function initLegend() {
            const legend = $('#legend');
            const allSpeakers = new Set();
            phase1Data.segments.forEach(seg => allSpeakers.add(seg.speaker));
            Object.keys(phase2Data.speakers || {}).forEach(speaker => allSpeakers.add(speaker));
            
            Array.from(allSpeakers).forEach((speaker, index) => {
                const item = $('<div>').addClass('legend-item');
                const colorBox = $('<div>').addClass('legend-color')
                    .css('background-color', colors[index % colors.length]);
                const label = $('<span>').text(speaker).addClass('small');
                
                item.append(colorBox).append(label);
                legend.append(item);
            });
        }

        function initControls() {
            $('#zoomLevel').on('input', function() {
                zoomLevel = parseInt($(this).val()) / 100;
                $('#zoomValue').text($(this).val() + '%');
                $('.timeline-scroll-content').css('transform', `scaleX(${zoomLevel})`);
                $('.timeline-scroll-content').css('transform-origin', 'left center');
            });
            
            $('#showPhase1').on('change', function() {
                $('.segment-phase1').toggle(this.checked);
            });
            
            $('#showPhase2').on('change', function() {
                $('.segment-phase2').toggle(this.checked);
            });
        }

        function showStatistics() {
            $('#statsPhase1').html(formatStatsReadable(phase1Data, 'Fase 1'));
            $('#statsPhase2').html(formatStatsReadable(phase2Data, 'Fase 2'));
        }

        function formatStatsReadable(phaseData, phaseName) {
            const meta = phaseData.metadata || {};
            const stats = meta.stats || {};
            const segments = phaseData.segments || [];
            
            let output = `=== ${phaseName} ===\n`;
            
            if (phaseName === 'Fase 1') {
                output += `Segmenti totali: ${segments.length}\n`;
                output += `Speaker univoci: ${new Set(segments.map(s => s.speaker)).size}\n`;
                
                if (meta.total_duration) {
                    output += `Durata audio: ${meta.total_duration.toFixed(1)}s\n`;
                }
                
                // Calcola statistiche durata segmenti
                if (segments.length > 0) {
                    const durations = segments.map(s => s.duration);
                    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
                    const minDuration = Math.min(...durations);
                    const maxDuration = Math.max(...durations);
                    
                    output += `Durata media segmenti: ${avgDuration.toFixed(2)}s\n`;
                    output += `Durata min: ${minDuration.toFixed(2)}s\n`;
                    output += `Durata max: ${maxDuration.toFixed(2)}s\n`;
                }
                
            } else if (phaseName === 'Fase 2') {
                output += `Segmenti iniziali: ${stats.total_segments_pre_merge || segments.length}\n`;
                output += `Segmenti dopo merge: ${stats.total_segments_post_merge || 'N/A'}\n`;
                output += `Segmenti uniti: ${stats.segments_merged || 'N/A'}\n`;
                output += `Segmenti rimossi: ${stats.short_segments_removed || 'N/A'}\n`;
                output += `Segmenti finali: ${stats.total_segments_final || segments.length}\n`;
                
                if (stats.avg_duration_all_speakers) {
                    output += `Durata media finale: ${stats.avg_duration_all_speakers.toFixed(2)}s\n`;
                }
                
                // Mostra parametri usati
                if (meta.parameters) {
                    output += `\nParametri filtro:\n`;
                    output += `- Min pause: ${meta.parameters.min_pause}s\n`;
                    output += `- Min duration: ${meta.parameters.min_duration}s\n`;
                }
            }
            
            return output;
        }

        function updateDebugInfo() {
            $('#debugInfo').html(`
                <strong>Debug Info:</strong><br>
                File: ${fileName}<br>
                Durata: ${waveformData.duration?.toFixed(1) || 0}s<br>
                Waveform points: ${waveformData.data?.length || 0}<br>
                Sample rate: ${sampleRate}Hz<br>
                Total pixels: ${waveformData.total_pixels}px<br>
                Timestamp: ogni 30 secondi
            `);
        }

        function formatTime(seconds) {
            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        // Inizializza quando il documento è pronto
        $(document).ready(initTimeline);
    </script>
</head>
<body>
    <div class="container-fluid my-4">
        <h1>[<strong>{{FILE_NAME}}</strong>] - Diarizzazione - Report post filtraggio</h1>
        
	<div class="controls mb-1 mt-4 d-flex">
	    <div class="border rounded p-2">
                <div class="legend" id="legend"></div>
	    </div>

	    <div class="p-1"></div>

	    <div class="border rounded p-1">
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="showPhase1" checked>
                    <label class="form-check-label" for="showPhase1">Mostra Fase 1</label>
                </div>
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="showPhase2" checked>
                    <label class="form-check-label" for="showPhase2">Mostra Fase 2</label>
                </div>
	    </div>

	    <div class="p-1"></div>

            <div class="border rounded p-1">
                <label for="zoomLevel" class="form-label">Zoom: <span id="zoomValue">100%</span></label>
                <input type="range" class="form-range" id="zoomLevel" min="10" max="200" value="100">
            </div>

	</div>

        <div class="timeline-container border rounded mb-1 mt-1">
            <div class="timeline-scroll-content" id="scrollContent">
                <div class="waveform-container" id="waveformContainer">
                    <!-- I segmenti della waveform verranno aggiunti qui -->
                </div>
                <div id="speakerTimelines"></div>
                <div class="time-markers" id="timeMarkers"></div>
            </div>
        </div>

        <div class="row stats-panel">
            <div class="col-md-6">
                <h5>Statistiche Fase 1</h5>
                <pre id="statsPhase1" class="bg-light p-3 small stats-readable"></pre>
            </div>
            <div class="col-md-6">
                <h5>Statistiche Fase 2</h5>
                <pre id="statsPhase2" class="bg-light p-3 small stats-readable"></pre>
            </div>
        </div>

        <!-- Debug info spostato in fondo -->
        <div class="debug-info" id="debugInfo">
            <strong>Debug Info:</strong> Caricamento in corso...
        </div>
    </div>
</body>
</html>
'''

def dump_template(output_path):
    """Salva il template corrente in un file"""
    template_content = get_fallback_template()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template_content)
    print(f"Template salvato in: {output_path}")

def find_template(template_path=None):
    """Cerca il template HTML"""
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Usa sempre il template integrato come fallback
    return get_fallback_template()

def generate_report(project_dir, output_path, sample_rate=10, template_path=None):
    """Genera il report HTML"""
    
    # Carica dati
    wav_path = get_wav_file_path(project_dir)
    if not wav_path:
        print("ERRORE: File WAV non trovato")
        return False
    
    phase1_path = common_utils.get_input_json_path(project_dir, [], "fase1_unified.json")
    phase2_path = common_utils.get_input_json_path(project_dir, [], "fase2_filtered.json")
    
    if not phase1_path or not phase2_path:
        print("ERRORE: File JSON fase1 o fase2 non trovati")
        return False
    
    print("Caricamento dati...")
    phase1_data = load_json_data(phase1_path)
    phase2_data = load_json_data(phase2_path)

    
    print("Campionamento waveform...")
    waveform_data = sample_audio_waveform(wav_path, sample_rate)
    if not waveform_data:
        return False
   
    # Filename
    source_file = phase1_data.get('metadata', {}).get('source_file', 'File sconosciuto')
    file_basename = os.path.basename(source_file)

    print("Preparazione template...")
    html_content = find_template(template_path)
    
    # Sostituisce i placeholder con i dati JSON
    html_content = html_content.replace('{{WAVEFORM_DATA}}', json.dumps(waveform_data))
    html_content = html_content.replace('{{PHASE1_DATA}}', json.dumps(phase1_data))
    html_content = html_content.replace('{{PHASE2_DATA}}', json.dumps(phase2_data))
    html_content = html_content.replace('{{SAMPLE_RATE}}', str(sample_rate))
    html_content = html_content.replace('{{FILE_NAME}}', file_basename)
    
    # Salva il report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generato: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="FASE 2 REPORT - Genera report HTML interattivo con waveform e timeline"
    )
    parser.add_argument("--project-dir", required=True, help="Directory del progetto")
    parser.add_argument("--output", default="fase2_report.html", help="File di output (default: fase2_report.html)")
    parser.add_argument("--sample-rate", type=float, default=10.0, 
                       help="Risoluzione campionamento waveform (Hz) - default: 10Hz = 0.1s per pixel")
    parser.add_argument("--template", help="Percorso template HTML personalizzato")
    parser.add_argument("--dump-template", help="Salva il template integrato in un file e esci")
    
    args = parser.parse_args()
    
    # Gestione --dump-template
    if args.dump_template:
        dump_template(args.dump_template)
        return 0
    
    if not os.path.exists(args.project_dir):
        print(f"ERRORE: Directory progetto non trovata: {args.project_dir}")
        return 1
    
    output_path = os.path.join(args.project_dir, args.output)
    
    success = generate_report(
        project_dir=args.project_dir,
        output_path=output_path,
        sample_rate=args.sample_rate,
        template_path=args.template
    )
    
    if success:
        print("✓ Report generato con successo")
        return 0
    else:
        print("✗ Errore nella generazione del report")
        return 1

if __name__ == "__main__":
    main()
