pipeline { //must be top-level
  
  agent any // where to execute. run on any/next available jenkins agent. agent can be a node
  
  stages { // where different stages happen. user to decide how many stages
  
    stage("build") { // usually build, test, deploy stage
    
      steps { // script that executes command on jenkins server/agent. e.g. npm install, npm build
        echo 'Building the application'
        echo 'add some change'
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
