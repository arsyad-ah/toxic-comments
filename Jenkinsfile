pipeline { //must be top-level
  
//   agent any // where to execute. run on any/next available jenkins agent. agent can be a node
  agent { docker { image 'continuumio/miniconda3:latest' }
        }
  stages { // where different stages happen. user to decide how many stages
  
    stage("build") { // usually build, test, deploy stage
      steps { // script that executes command on jenkins server/agent. e.g. npm install, npm build
        echo 'Building the application'
        sh '''
          apt-get update && apt-get install build-essential -y
          conda env update --file conda.yml'
          conda activate toxic-clf
          '''
//         sh 'pylint -E src'
        echo 'pylint completed!'
      }    
    }
    
    stage("test") { // usually build, test, deploy stage
    
      steps { // script that executes command on jenkins server/agent. e.g. npm install, npm build
        echo 'Testing the application'
      }
    }
    
    stage("deploy") { // usually build, test, deploy stage
    
      steps { // script that executes command on jenkins server/agent. e.g. npm install, npm build
        echo 'Deploying the application'
      }  
    }
  }
}

// pipeline {
//     agent {
//         docker { image 'node:14-alpine' }
//     }
//     stages {
//         stage('Test') {
//             steps {
//                 sh 'node --version'
//             }
//         }
//     }
// }
