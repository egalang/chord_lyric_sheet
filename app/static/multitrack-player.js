class StemSyncPlayer {
  constructor(root) {
    this.root = root;
    this.ctx = null;
    this.masterGain = null;
    this.tracks = [];
    this.isPlaying = false;
    this.startedAt = 0;
    this.pausedAt = 0;
    this.duration = 0;
    this.animationFrame = null;
    this.resizeHandler = null;

    this.playButton = root.querySelector('[data-multitrack-play]');
    this.seekInput = root.querySelector('[data-multitrack-seek]');
    this.timeLabel = root.querySelector('[data-multitrack-time]');
    this.tracksHost = root.querySelector('[data-multitrack-tracks]');
    this.emptyState = root.querySelector('[data-multitrack-empty]');

    this.bindEvents();

    this.trackColors = [
      '#2563eb', // blue
      '#7c3aed', // purple
      '#10b981', // green
      '#f59e0b', // orange
      '#ef4444', // red
      '#06b6d4', // cyan
    ];
  }

  bindEvents() {
    if (this.playButton) {
      this.playButton.addEventListener('click', () => {
        if (!this.tracks.length) return;
        if (this.isPlaying) {
          this.pause();
        } else {
          this.play();
        }
      });
    }

    if (this.seekInput) {
      this.seekInput.addEventListener('input', (event) => {
        const ratio = Number(event.target.value || 0) / 1000;
        const nextTime = ratio * this.duration;
        this.seek(nextTime);
      });
    }
  }

  async ensureContext() {
    if (!this.ctx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioContextClass();
      this.masterGain = this.ctx.createGain();
      this.masterGain.connect(this.ctx.destination);
    }

    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
  }

  async load(manifest) {
    await this.ensureContext();
    this.stop(true);
    this.clearTracks();

    const stems = Array.isArray(manifest?.stems) ? manifest.stems : [];
    if (!stems.length) {
      this.showEmpty('No stems available yet.');
      return;
    }

    this.showEmpty('Loading stems...');

    const decodedTracks = [];
    let index = 0;

    for (const stem of stems) {
      const response = await fetch(stem.url);
      if (!response.ok) {
        throw new Error(`Failed to load stem: ${stem.label}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.ctx.decodeAudioData(arrayBuffer.slice(0));

      const gainNode = this.ctx.createGain();
      gainNode.gain.value = 1;
      gainNode.connect(this.masterGain);

      decodedTracks.push({
        id: stem.id,
        label: stem.label,
        url: stem.url,
        buffer: audioBuffer,
        gainNode,
        sourceNode: null,
        muted: false,
        row: null,
        canvas: null,
        muteButton: null,
        color: this.trackColors[index % this.trackColors.length],
        icon: this.iconForStem(stem),
      });

      index += 1;
    }

    this.tracks = decodedTracks;
    this.duration = Math.max(...decodedTracks.map((track) => track.buffer.duration));
    this.pausedAt = 0;

    this.renderTracks();
    this.updateTimeline();
    this.hideEmpty();
  }

  clearTracks() {
    this.stopTick();
    if (this.resizeHandler) {
      window.removeEventListener('resize', this.resizeHandler);
      this.resizeHandler = null;
    }
    if (this.tracksHost) {
      this.tracksHost.innerHTML = '';
    }
    this.tracks = [];
    this.duration = 0;
  }

  formatTime(seconds) {
    const value = Math.max(0, Math.floor(seconds || 0));
    const mins = Math.floor(value / 60);
    const secs = value % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
  }

  showEmpty(message) {
    if (this.emptyState) {
      this.emptyState.hidden = false;
      this.emptyState.textContent = message;
    }
  }

  hideEmpty() {
    if (this.emptyState) {
      this.emptyState.hidden = true;
    }
  }

  iconForStem(stem) {
    const key = `${stem?.id || ''} ${stem?.label || ''}`.toLowerCase();
    if (key.includes('vocal') || key.includes('voice')) return '🎤';
    if (key.includes('drum') || key.includes('perc')) return '🥁';
    if (key.includes('bass')) return '🎸';
    if (key.includes('piano') || key.includes('keys') || key.includes('keyboard')) return '🎹';
    if (key.includes('guitar')) return '🎸';
    return '🎵';
  }

  createSource(track, offsetSeconds) {
    const source = this.ctx.createBufferSource();
    source.buffer = track.buffer;
    source.connect(track.gainNode);
    source.start(0, Math.min(offsetSeconds, track.buffer.duration));
    return source;
  }

  play() {
    if (!this.ctx || this.isPlaying || !this.tracks.length) return;

    this.startedAt = this.ctx.currentTime - this.pausedAt;
    for (const track of this.tracks) {
      track.sourceNode = this.createSource(track, this.pausedAt);
    }

    this.isPlaying = true;
    if (this.playButton) this.playButton.textContent = 'Pause';
    this.tick();
  }

  pause() {
    if (!this.isPlaying) return;
    this.pausedAt = this.currentTime();
    this.stopSources();
    this.isPlaying = false;
    if (this.playButton) this.playButton.textContent = 'Play';
    this.stopTick();
    this.updateTimeline();
  }

  stop(resetPosition = false) {
    this.stopSources();
    this.isPlaying = false;
    if (resetPosition) {
      this.pausedAt = 0;
    }
    if (this.playButton) this.playButton.textContent = 'Play';
    this.stopTick();
    this.updateTimeline();
  }

  stopSources() {
    for (const track of this.tracks) {
      if (track.sourceNode) {
        try {
          track.sourceNode.stop();
        } catch (error) {
          // Ignore sources already stopped.
        }
        track.sourceNode.disconnect();
        track.sourceNode = null;
      }
    }
  }

  currentTime() {
    if (!this.ctx) return 0;
    if (!this.isPlaying) return this.pausedAt;
    return Math.min(this.ctx.currentTime - this.startedAt, this.duration || 0);
  }

  seek(seconds) {
    this.pausedAt = Math.max(0, Math.min(seconds, this.duration || 0));
    if (this.isPlaying) {
      this.stopSources();
      this.startedAt = this.ctx.currentTime - this.pausedAt;
      for (const track of this.tracks) {
        track.sourceNode = this.createSource(track, this.pausedAt);
      }
    }
    this.updateTimeline();
  }

  tick() {
    this.updateTimeline();
    if (this.currentTime() >= this.duration) {
      this.stop(true);
      return;
    }
    this.animationFrame = window.requestAnimationFrame(() => this.tick());
  }

  stopTick() {
    if (this.animationFrame) {
      window.cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  updateTimeline() {
    const now = this.currentTime();
    if (this.timeLabel) {
      this.timeLabel.textContent = `${this.formatTime(now)} / ${this.formatTime(this.duration)}`;
    }
    if (this.seekInput) {
      const ratio = this.duration ? now / this.duration : 0;
      this.seekInput.value = String(Math.round(ratio * 1000));
    }

    for (const track of this.tracks) {
      this.drawWaveform(track, now);
      this.syncMuteButton(track);
    }
  }

  syncMuteButton(track) {
    const button = track.muteButton;
    if (!button) return;
    const icon = track.muted ? '🔇' : '🔊';
    const action = track.muted ? 'Unmute' : 'Mute';
    button.textContent = icon;
    button.setAttribute('aria-label', `${action} ${track.label}`);
    button.setAttribute('title', `${action} ${track.label}`);
    button.classList.toggle('is-muted', track.muted);
  }

  toggleMute(track) {
    track.muted = !track.muted;
    track.gainNode.gain.value = track.muted ? 0 : 1;
    this.syncMuteButton(track);
  }

  renderTracks() {
    if (!this.tracksHost) return;
    this.tracksHost.innerHTML = '';

    for (const track of this.tracks) {
      const row = document.createElement('div');
      row.className = 'multitrack-row';
      row.style.setProperty('--track-color', track.color);

      const meta = document.createElement('div');
      meta.className = 'multitrack-row-meta';

      const labelWrap = document.createElement('div');
      labelWrap.className = 'multitrack-track-heading';

      const labelIcon = document.createElement('span');
      labelIcon.className = 'multitrack-track-icon';
      labelIcon.textContent = track.icon;
      labelIcon.setAttribute('aria-hidden', 'true');

      const label = document.createElement('div');
      label.className = 'multitrack-track-label';
      label.textContent = track.label;

      labelWrap.appendChild(labelIcon);
      labelWrap.appendChild(label);

      const controls = document.createElement('div');
      controls.className = 'multitrack-track-controls';

      const muteButton = document.createElement('button');
      muteButton.type = 'button';
      muteButton.className = 'multitrack-track-button';
      muteButton.setAttribute('data-track-mute', '');
      muteButton.addEventListener('click', () => this.toggleMute(track));

      controls.appendChild(muteButton);
      meta.appendChild(labelWrap);
      meta.appendChild(controls);

      const lane = document.createElement('div');
      lane.className = 'multitrack-wave-lane';

      const canvas = document.createElement('canvas');
      canvas.className = 'multitrack-wave-canvas';
      lane.appendChild(canvas);

      row.appendChild(meta);
      row.appendChild(lane);

      track.row = row;
      track.canvas = canvas;
      track.muteButton = muteButton;
      this.syncMuteButton(track);
      this.tracksHost.appendChild(row);
    }

    const resizeAll = () => {
      for (const track of this.tracks) {
        this.drawWaveform(track, this.currentTime());
      }
    };

    if (this.resizeHandler) {
      window.removeEventListener('resize', this.resizeHandler);
    }
    this.resizeHandler = resizeAll;
    window.requestAnimationFrame(resizeAll);
    window.addEventListener('resize', this.resizeHandler, { passive: true });
  }

  drawWaveform(track, playheadSeconds = 0) {
    const canvas = track.canvas;
    if (!canvas) return;

    const lane = canvas.parentElement;
    const width = Math.max(400, Math.floor(lane.clientWidth || 400));
    const height = 84;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    const ctx = canvas.getContext('2d');
    const buffer = track.buffer;
    const data = buffer.getChannelData(0);
    const step = Math.max(1, Math.floor(data.length / width));
    const amp = height / 2;

    const styles = getComputedStyle(document.documentElement);
    const bg = styles.getPropertyValue('--surface-2').trim() || '#f8fbff';

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = track.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < width; i += 1) {
      let min = 1;
      let max = -1;
      const start = i * step;
      const end = Math.min(start + step, data.length);
      for (let j = start; j < end; j += 1) {
        const value = data[j];
        if (value < min) min = value;
        if (value > max) max = value;
      }
      ctx.moveTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }
    ctx.stroke();

    const playheadRatio = this.duration ? playheadSeconds / this.duration : 0;
    const x = Math.max(0, Math.min(width, Math.floor(playheadRatio * width)));

    ctx.fillStyle = `${track.color}22`;
    ctx.fillRect(0, 0, x, height);

    ctx.strokeStyle = track.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

async function loadMultitrackIntoPage(manifestUrl) {
  const root = document.querySelector('[data-multitrack-player]');
  if (!root) return null;

  if (!window.__stemSyncPlayer) {
    window.__stemSyncPlayer = new StemSyncPlayer(root);
  }

  if (!manifestUrl) {
    window.__stemSyncPlayer.showEmpty('No stem manifest available yet.');
    return window.__stemSyncPlayer;
  }

  const response = await fetch(manifestUrl);
  if (!response.ok) {
    throw new Error('Unable to load stem manifest.');
  }
  const manifest = await response.json();
  await window.__stemSyncPlayer.load(manifest);
  return window.__stemSyncPlayer;
}

window.loadMultitrackIntoPage = loadMultitrackIntoPage;
