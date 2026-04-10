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

    this.playButton = root.querySelector('[data-multitrack-play]');
    this.seekInput = root.querySelector('[data-multitrack-seek]');
    this.timeLabel = root.querySelector('[data-multitrack-time]');
    this.tracksHost = root.querySelector('[data-multitrack-tracks]');
    this.emptyState = root.querySelector('[data-multitrack-empty]');

    this.bindEvents();

    // 🎨 DAW-style track colors
    this.trackColors = [
      '#2563eb', // blue
      '#7c3aed', // purple
      '#10b981', // green
      '#f59e0b', // orange
      '#ef4444', // red
      '#06b6d4'  // cyan
    ];
  }

  bindEvents() {
    if (this.playButton) {
      this.playButton.addEventListener('click', () => {
        if (!this.tracks.length) return;
        if (this.isPlaying) this.pause();
        else this.play();
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
      if (!response.ok) throw new Error(`Failed to load stem: ${stem.label}`);

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
        color: this.trackColors[index % this.trackColors.length],
      });

      index++;
    }

    this.tracks = decodedTracks;
    this.duration = Math.max(...decodedTracks.map(t => t.buffer.duration));
    this.pausedAt = 0;

    this.renderTracks();
    this.updateTimeline();
    this.hideEmpty();
  }

  clearTracks() {
    this.tracksHost.innerHTML = '';
    this.tracks = [];
    this.duration = 0;
  }

  formatTime(seconds) {
    const v = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(v / 60)}:${String(v % 60).padStart(2, '0')}`;
  }

  showEmpty(msg) {
    if (this.emptyState) {
      this.emptyState.hidden = false;
      this.emptyState.textContent = msg;
    }
  }

  hideEmpty() {
    if (this.emptyState) this.emptyState.hidden = true;
  }

  createSource(track, offset) {
    const source = this.ctx.createBufferSource();
    source.buffer = track.buffer;
    source.connect(track.gainNode);
    source.start(0, Math.min(offset, track.buffer.duration));
    return source;
  }

  play() {
    if (!this.ctx || this.isPlaying || !this.tracks.length) return;

    this.startedAt = this.ctx.currentTime - this.pausedAt;

    for (const track of this.tracks) {
      track.sourceNode = this.createSource(track, this.pausedAt);
    }

    this.isPlaying = true;
    this.playButton.textContent = 'Pause';
    this.tick();
  }

  pause() {
    if (!this.isPlaying) return;

    this.pausedAt = this.currentTime();
    this.stopSources();
    this.isPlaying = false;
    this.playButton.textContent = 'Play';
    this.stopTick();
    this.updateTimeline();
  }

  stop(reset = false) {
    this.stopSources();
    this.isPlaying = false;
    if (reset) this.pausedAt = 0;
    this.playButton.textContent = 'Play';
    this.stopTick();
    this.updateTimeline();
  }

  stopSources() {
    for (const t of this.tracks) {
      if (t.sourceNode) {
        try { t.sourceNode.stop(); } catch {}
        t.sourceNode.disconnect();
        t.sourceNode = null;
      }
    }
  }

  currentTime() {
    if (!this.ctx) return 0;
    if (!this.isPlaying) return this.pausedAt;
    return Math.min(this.ctx.currentTime - this.startedAt, this.duration);
  }

  seek(sec) {
    this.pausedAt = Math.max(0, Math.min(sec, this.duration));

    if (this.isPlaying) {
      this.stopSources();
      this.startedAt = this.ctx.currentTime - this.pausedAt;

      for (const t of this.tracks) {
        t.sourceNode = this.createSource(t, this.pausedAt);
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

    this.animationFrame = requestAnimationFrame(() => this.tick());
  }

  stopTick() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  updateTimeline() {
    const now = this.currentTime();

    if (this.timeLabel) {
      this.timeLabel.textContent =
        `${this.formatTime(now)} / ${this.formatTime(this.duration)}`;
    }

    if (this.seekInput) {
      const ratio = this.duration ? now / this.duration : 0;
      this.seekInput.value = String(Math.round(ratio * 1000));
    }

    for (const track of this.tracks) {
      this.drawWaveform(track, now);
    }
  }

  toggleMute(track) {
    track.muted = !track.muted;
    track.gainNode.gain.value = track.muted ? 0 : 1;

    const btn = track.row?.querySelector('[data-track-mute]');
    if (btn) {
      btn.textContent = track.muted ? 'Unmute' : 'Mute';
      btn.classList.toggle('is-muted', track.muted);
    }
  }

  renderTracks() {
    this.tracksHost.innerHTML = '';

    for (const track of this.tracks) {
      const row = document.createElement('div');
      row.className = 'multitrack-row';

      const meta = document.createElement('div');
      meta.className = 'multitrack-row-meta';

      const label = document.createElement('div');
      label.className = 'multitrack-track-label';
      label.textContent = track.label;

      const muteBtn = document.createElement('button');
      muteBtn.className = 'multitrack-track-button';
      muteBtn.textContent = 'Mute';
      muteBtn.onclick = () => this.toggleMute(track);

      meta.appendChild(label);
      meta.appendChild(muteBtn);

      const lane = document.createElement('div');
      lane.className = 'multitrack-wave-lane';

      const canvas = document.createElement('canvas');
      canvas.className = 'multitrack-wave-canvas';
      lane.appendChild(canvas);

      row.appendChild(meta);
      row.appendChild(lane);

      track.row = row;
      track.canvas = canvas;

      this.tracksHost.appendChild(row);
    }
  }

  drawWaveform(track, playheadSeconds = 0) {
    const canvas = track.canvas;
    if (!canvas) return;

    const lane = canvas.parentElement;
    const width = Math.max(400, lane.clientWidth || 400);
    const height = 84;

    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    const ctx = canvas.getContext('2d');
    const buffer = track.buffer;
    const data = buffer.getChannelData(0);

    const step = Math.floor(data.length / width);
    const amp = height / 2;

    const styles = getComputedStyle(document.documentElement);
    const bg = styles.getPropertyValue('--surface-2').trim();

    ctx.clearRect(0, 0, width, height);

    // background
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    // waveform (per-track color)
    ctx.strokeStyle = track.color;
    ctx.lineWidth = 1;

    ctx.beginPath();
    for (let i = 0; i < width; i++) {
      let min = 1, max = -1;
      const start = i * step;
      const end = start + step;

      for (let j = start; j < end; j++) {
        const val = data[j] || 0;
        if (val < min) min = val;
        if (val > max) max = val;
      }

      ctx.moveTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }
    ctx.stroke();

    const x = (playheadSeconds / this.duration) * width;

    // played region
    ctx.fillStyle = track.color + '22';
    ctx.fillRect(0, 0, x, height);

    // playhead
    ctx.strokeStyle = track.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

window.loadMultitrackIntoPage = async function (manifestUrl) {
  const root = document.querySelector('[data-multitrack-player]');
  if (!root) return null;

  if (!window.__stemSyncPlayer) {
    window.__stemSyncPlayer = new StemSyncPlayer(root);
  }

  if (!manifestUrl) {
    window.__stemSyncPlayer.showEmpty('No stem manifest available yet.');
    return window.__stemSyncPlayer;
  }

  const res = await fetch(manifestUrl);
  const manifest = await res.json();
  await window.__stemSyncPlayer.load(manifest);

  return window.__stemSyncPlayer;
};