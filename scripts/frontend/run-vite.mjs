#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const MIN_NODE_MAJOR = 18
const REEXEC_ENV = 'PIERN_VITE_NODE_REEXEC'

const scriptPath = fileURLToPath(import.meta.url)
const projectRoot = resolve(dirname(scriptPath), '../..')
const viteBin = join(projectRoot, 'frontend', 'node_modules', 'vite', 'bin', 'vite.js')
const args = process.argv.slice(2)

function nodeMajor(versionText) {
  const match = versionText.trim().match(/^v?(\d+)/)
  return match ? Number(match[1]) : 0
}

function currentNodeMajor() {
  return nodeMajor(process.versions.node)
}

function candidateNodes() {
  const seen = new Set()
  const nodes = []
  const add = (candidate) => {
    if (!candidate || seen.has(candidate)) return
    seen.add(candidate)
    nodes.push(candidate)
  }

  add(process.env.PIERN_NODE)
  if (process.env.CONDA_PREFIX) add(join(process.env.CONDA_PREFIX, 'bin', 'node'))
  if (process.env.HOME) {
    add(join(process.env.HOME, '.conda', 'envs', 'piern', 'bin', 'node'))
    add(join(process.env.HOME, 'miniconda3', 'envs', 'piern', 'bin', 'node'))
    add(join(process.env.HOME, 'anaconda3', 'envs', 'piern', 'bin', 'node'))
  }
  return nodes
}

function compatibleNode(candidate) {
  if (!existsSync(candidate)) return false
  const result = spawnSync(candidate, ['--version'], { encoding: 'utf8' })
  if (result.status !== 0) return false
  return nodeMajor(result.stdout) >= MIN_NODE_MAJOR
}

function reexecWithCompatibleNode() {
  for (const candidate of candidateNodes()) {
    if (!compatibleNode(candidate)) continue
    const result = spawnSync(candidate, [scriptPath, ...args], {
      stdio: 'inherit',
      env: {
        ...process.env,
        [REEXEC_ENV]: '1',
        PATH: `${dirname(candidate)}:${process.env.PATH ?? ''}`,
      },
    })
    if (result.error) {
      console.error(result.error.message)
      process.exit(1)
    }
    process.exit(result.status ?? 1)
  }

  console.error(
    `Vite 需要 Node ${MIN_NODE_MAJOR}+，当前是 ${process.version}。` +
    ' 请设置 PIERN_NODE=/path/to/node，或激活包含新版 Node 的 piern conda 环境。',
  )
  process.exit(1)
}

if (currentNodeMajor() < MIN_NODE_MAJOR) {
  if (process.env[REEXEC_ENV]) {
    console.error(`Vite 需要 Node ${MIN_NODE_MAJOR}+，当前是 ${process.version}。`)
    process.exit(1)
  }
  reexecWithCompatibleNode()
}

if (!globalThis.crypto?.getRandomValues) {
  const { webcrypto } = await import('node:crypto')
  globalThis.crypto = webcrypto
}

await import(pathToFileURL(viteBin).href)
