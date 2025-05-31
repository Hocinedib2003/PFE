// const { app, BrowserWindow } = require('electron')
// const path = require('path')
// const { spawn } = require('child_process')
//
// let mainWindow
// let pythonProcess
//
// function createWindow() {
//   mainWindow = new BrowserWindow({
//     width: 1200,
//     height: 800,
//     webPreferences: {
//       nodeIntegration: false,
//       contextIsolation: true,
//       webSecurity: false
//     }
//   })
//
//   // Chemin absolu vers le dossier public
//   const appPath= path.join(__dirname, 'public', 'application.py')
//
//   // Lance Streamlit
//   pythonProcess = spawn('python', [
//     '-m',
//     'streamlit',
//     'run',
//     appPath,
//     '--server.port=8501',
//     '--server.headless=true',
//     '--browser.serverAddress=localhost'
//   ], {
//     stdio: 'pipe'
//   })
//
//   // Debug - Affiche les logs dans la console
//   pythonProcess.stdout.on('data', (data) => {
//     console.log(`Streamlit: ${data}`)
//   })
//   pythonProcess.stderr.on('data', (data) => {
//     console.error(`Streamlit ERROR: ${data}`)
//   })
//
//   // Charge l'interface après 3s (temps de démarrage Streamlit)
//   setTimeout(() => {
//     mainWindow.loadURL('http://localhost:8501')
//   }, 3000)
// }
//
// app.whenReady().then(createWindow)
//
// // Fermeture propre
// app.on('window-all-closed', () => {
//   if (pythonProcess) pythonProcess.kill()
//   app.quit()
// })

const { app, BrowserWindow } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const fs = require('fs')

let mainWindow
let pythonProcess

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false
    }
  })

  // Chemin différent en développement vs production
  let appPath
  if (app.isPackaged) {
    // En production, les fichiers sont dans resources/app/public
    appPath = path.join(process.resourcesPath, 'app', 'public', 'application.py')
  } else {
    // En développement, chemin normal
    appPath = path.join(__dirname, 'public', 'application.py')
  }

  // Vérifie que le fichier existe
  if (!fs.existsSync(appPath)) {
    console.error('Fichier application.py introuvable:', appPath)
    app.quit()
    return
  }

  // Lance Streamlit
  pythonProcess = spawn('python', [
    '-m',
    'streamlit',
    'run',
    appPath,
    '--server.port=8501',
    '--server.headless=true',
    '--browser.serverAddress=localhost'
  ], {
    stdio: 'pipe'
  })

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Streamlit: ${data}`)
  })
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Streamlit ERROR: ${data}`)
  })

  // Charge l'interface après 3s
  setTimeout(() => {
    mainWindow.loadURL('http://localhost:8501')
  }, 3000)
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill()
  app.quit()
})