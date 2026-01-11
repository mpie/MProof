/* eslint-disable no-console */
const { execSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function getLightningCssVersion() {
  const entry = require.resolve('lightningcss'); // resolves to lightningcss/node/index.js
  const pkgPath = path.resolve(entry, '..', '..', 'package.json');
  const raw = fs.readFileSync(pkgPath, 'utf8');
  const pkg = JSON.parse(raw);
  return pkg.version;
}

function getTargetPackage() {
  const platform = process.platform;
  const arch = process.arch;

  if (platform === 'darwin') return `lightningcss-darwin-${arch}`;
  if (platform === 'win32') return `lightningcss-win32-${arch}-msvc`;

  if (platform === 'linux') {
    // Keep it simple: most local dev is glibc.
    // If you run on musl (Alpine), reinstall deps in that environment.
    return `lightningcss-linux-${arch}-gnu`;
  }

  return null;
}

function ensure() {
  const target = getTargetPackage();
  if (!target) return;

  try {
    // eslint-disable-next-line import/no-extraneous-dependencies
    require(target);
    return;
  } catch {
    // continue
  }

  const version = getLightningCssVersion();
  console.log(`[ensure-lightningcss] Missing ${target}. Installing ${target}@${version}...`);

  execSync(
    `npm install --no-save --no-package-lock --no-audit --no-fund ${target}@${version}`,
    { stdio: 'inherit' }
  );

  // Verify lightningcss itself loads (it will require the right native binding).
  // eslint-disable-next-line import/no-extraneous-dependencies
  require('lightningcss');
  console.log('[ensure-lightningcss] lightningcss OK');
}

ensure();

