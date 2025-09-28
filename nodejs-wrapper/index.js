/**
 * Programmatic API for minimal pairs audio tools
 * This provides a Node.js API wrapper around the Python CLI
 */

import { execa } from 'execa';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check if we're in development or installed
const isDevelopment = existsSync(path.join(__dirname, '../src'));
const audioToolsPath = isDevelopment
  ? path.join(__dirname, '..')  // Parent directory is the audio_tools root
  : path.join(__dirname, 'audio_tools');

/**
 * Base class for audio tools operations
 */
class AudioTools {
  constructor(options = {}) {
    this.audioToolsPath = options.audioToolsPath || audioToolsPath;
    this.verbose = options.verbose || false;
  }

  /**
   * Execute a command with the audio tools CLI
   */
  async _execute(command, args = [], options = {}) {
    const uvArgs = [
      'tool', 'run',
      '--from', '.',
      'minimal-pairs-audio',
      command,
      ...args
    ];

    if (this.verbose) {
      console.log(`Executing: uv ${uvArgs.join(' ')}`);
    }

    try {
      const result = await execa('uv', uvArgs, {
        cwd: this.audioToolsPath,
        env: {
          ...process.env,
          PYTHONPATH: path.join(this.audioToolsPath, 'src')
        },
        ...options
      });

      return {
        success: true,
        stdout: result.stdout,
        stderr: result.stderr,
        exitCode: result.exitCode
      };
    } catch (error) {
      return {
        success: false,
        stdout: error.stdout || '',
        stderr: error.stderr || error.message,
        exitCode: error.exitCode || 1
      };
    }
  }

  /**
   * Generate audio files
   */
  async generate(options = {}) {
    const args = [];

    if (options.outputDir) args.push('--output-dir', options.outputDir);
    if (options.overwrite) args.push('--overwrite');
    if (options.limitVoices) args.push('--limit-voices', options.limitVoices.toString());
    if (options.voiceType) args.push('--voice-type', options.voiceType);
    if (options.volumeGain) args.push('--volume-gain', options.volumeGain.toString());
    if (options.pairsFile) args.push('--pairs-file', options.pairsFile);

    return this._execute('generate', args);
  }

  /**
   * Verify audio quality
   */
  async verify(options = {}) {
    const args = [];

    if (options.model) args.push('--model', options.model);
    if (options.words) {
      options.words.forEach(word => args.push('--words', word));
    }
    if (options.category) args.push('--category', options.category);
    if (options.maxFiles) args.push('--max-files', options.maxFiles.toString());
    if (options.output) args.push('--output', options.output);
    if (options.pairsFile) args.push('--pairs-file', options.pairsFile);
    if (options.manifestFile) args.push('--manifest-file', options.manifestFile);

    return this._execute('verify', args);
  }

  /**
   * Generate manifest
   */
  async manifest(options = {}) {
    const args = [];

    if (options.audioDir) args.push('--audio-dir', options.audioDir);
    if (options.output) args.push('--output', options.output);
    if (options.verifyFiles) args.push('--verify-files');
    if (options.fixMissing) args.push('--fix-missing');

    return this._execute('manifest', args);
  }

  /**
   * Run full pipeline
   */
  async fullPipeline(options = {}) {
    const args = [];

    if (options.outputDir) args.push('--output-dir', options.outputDir);
    if (options.overwrite) args.push('--overwrite');
    if (options.limitVoices) args.push('--limit-voices', options.limitVoices.toString());
    if (options.voiceType) args.push('--voice-type', options.voiceType);
    if (options.volumeGain) args.push('--volume-gain', options.volumeGain.toString());
    if (options.pairsFile) args.push('--pairs-file', options.pairsFile);
    if (options.model) args.push('--model', options.model);
    if (options.maxVerifyFiles) args.push('--max-verify-files', options.maxVerifyFiles.toString());
    if (options.skipVerification) args.push('--skip-verification');

    return this._execute('full-pipeline', args);
  }

  /**
   * Regenerate audio for specific word
   */
  async regenerate(options = {}) {
    const args = [];

    if (!options.word) {
      throw new Error('word option is required for regenerate');
    }

    args.push('--word', options.word);
    if (options.voice) args.push('--voice', options.voice);
    if (options.outputDir) args.push('--output-dir', options.outputDir);
    if (options.volumeGain) args.push('--volume-gain', options.volumeGain.toString());
    if (options.pairsFile) args.push('--pairs-file', options.pairsFile);

    return this._execute('regenerate', args);
  }

  /**
   * Clean invalid audio files
   */
  async clean(options = {}) {
    const args = [];

    if (options.minSize) args.push('--min-size', options.minSize.toString());
    if (options.audioDir) args.push('--audio-dir', options.audioDir);
    if (options.dryRun) args.push('--dry-run');

    return this._execute('clean', args);
  }
}

// Export the class and a default instance
export { AudioTools };
export default new AudioTools();
