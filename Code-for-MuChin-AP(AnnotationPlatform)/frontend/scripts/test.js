'use strict'

process.env.BABEL_ENV = 'test'
process.env.NODE_ENV = 'test'
process.env.PUBLIC_URL = ''

process.on('unhandledRejection', (err) => {
  throw err
})

require('../config/env')

const jest = require('jest')
const execSync = require('child_process').execSync

let argv = process.argv.slice(2)

function isInGitRepository() {
  try {
    execSync('git rev-parse --is-inside-work-tree', { stdio: 'ignore' })
    return true
  } catch (e) {
    return false
  }
}

function isInMercurialRepository() {
  try {
    execSync('hg --cwd . root', { stdio: 'ignore' })
    return true
  } catch (e) {
    return false
  }
}

if (
  !process.env.CI &&
  argv.indexOf('--watchAll') === -1 &&
  argv.indexOf('--watchAll=false') === -1
) {
  const hasSourceControl = isInGitRepository() || isInMercurialRepository()
  argv.push(hasSourceControl ? '--watch' : '--watchAll')
}

jest.run(argv)