#!/usr/bin/env node

/**
 * Node.js CLI wrapper for minimal-pairs audio tools
 * This wraps the Python uv-based CLI and makes it available via npm/npx
 */

import { execa } from 'execa';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check if we're in development (running from source) or installed
const isDevelopment = existsSync(path.join(__dirname, '../../src'));
const audioToolsPath = isDevelopment
  ? path.join(__dirname, '../..')  // We're inside audio_tools/nodejs-wrapper/bin
  : path.join(__dirname, '../audio_tools');

async function main() {
  const args = process.argv.slice(2);

  // If no arguments, show help
  if (args.length === 0) {
    args.push('--help');
  }

  try {
    // Check if uv is available
    try {
      await execa('uv', ['--version'], { stdio: 'ignore' });
    } catch (error) {
      console.error('❌ Error: uv tool is not installed or not in PATH');
      console.error('');
      console.error('Please install uv first:');
      console.error('  curl -LsSf https://astral.sh/uv/install.sh | sh');
      console.error('');
      console.error('Or visit: https://github.com/astral-sh/uv');
      process.exit(1);
    }

    // Check if audio tools are available
    if (!existsSync(audioToolsPath)) {
      console.error('❌ Error: Audio tools not found');
      console.error(`Expected path: ${audioToolsPath}`);
      console.error('');
      console.error('This package requires the Python audio tools to be installed.');
      console.error('Please check the installation instructions.');
      process.exit(1);
    }

    // Build the command
    const uvArgs = [
      'tool', 'run',
      '--from', '.',
      'minimal-pairs-audio',
      ...args
    ];

    console.log(`🎵 Running: uv ${uvArgs.join(' ')}`);
    console.log(`📁 Working directory: ${audioToolsPath}`);
    console.log('');

    // Execute the command
    const subprocess = execa('uv', uvArgs, {
      cwd: audioToolsPath,
      stdio: 'inherit',
      env: {
        ...process.env,
        PYTHONPATH: path.join(audioToolsPath, 'src')
      }
    });

    // Handle process termination
    process.on('SIGINT', () => {
      subprocess.kill('SIGINT');
    });

    process.on('SIGTERM', () => {
      subprocess.kill('SIGTERM');
    });

    const { exitCode } = await subprocess;
    process.exit(exitCode || 0);

  } catch (error) {
    if (error.exitCode) {
      // Command failed, but this is expected (e.g., validation errors)
      process.exit(error.exitCode);
    } else {
      // Unexpected error
      console.error('❌ Unexpected error:', error.message);
      process.exit(1);
    }
  }
}

main();
