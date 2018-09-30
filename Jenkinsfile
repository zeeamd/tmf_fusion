#!/usr/bin/env groovy

import groovy.json.JsonOutput
//import junitxml

def slackNotificationChannel = "tmf_jenkins"

def notifySlack(text, channel, attachments) {
    def slackURL = 'https://hooks.slack.com/services/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'
    def jenkinsIcon = 'https://wiki.jenkins-ci.org/download/attachments/2916393/logo.png'

    def payload = JsonOutput.toJson([text: text,
                                     channel: channel,
                                     username: "Jenkins",
                                     icon_url: jenkinsIcon,
                                     attachments: attachments
    ])

    sh "curl -X POST --data-urlencode \'payload=${payload}\' ${slackURL}"
}




pipeline {
    agent {
        label 'build-agent'
    }
    parameters {
        booleanParam(defaultValue: true, description: 'Is Artifactory upload required?', name: 'artifactory_upload')
        booleanParam(defaultValue: false, description: 'Is S3 upload required?', name: 's3_deploy')
    }
    environment {
        JAVA_HOME='/mv_data/apps/jdk1.8'
    }
    stages {
        stage("Initial Setup") {
            steps{
                 echo "Starting with Build"
                notifySlack("", slackNotificationChannel, [
                        [
                                "title": "${env.JOB_NAME}, build #${env.BUILD_NUMBER}",
                                "title_link": "${env.BUILD_URL}",
                                "color": "#FFFF00",
                                "text": "Build Started!",
                                "fields": [
                                        [
                                                "title": "Branch",
                                                "value": "${env.GIT_BRANCH}",
                                                "short": true
                                        ]
                                ]
                        ]
                ])
            }
        }
        stage('Compile') {
            steps {
                sh "ls -la"
                sh "chmod a+x mvnw"
//                sh "./mvnw compile -Dmaven.test.skip=true"
                echo "Build is complete"
            }
        }
        stage('Unit test') {
            steps {
                sh "./mvnw -fn test"
                junit "**/**/target/surefire-reports/*.xml"
                echo "unit test is complete"
            }
    }
//        stage('Integration test') {
//            steps {
//                echo "Integration test is yet to implemented"
//            }
//}
stage('Upload to Artifactory') {
    when {
        expression {
            params.artifactory_upload == true
        }
    }
    steps {
        sh './mvnw deploy -Dmaven.test.skip=true'
        echo "Upload is completed"
    }
}
//stage('Deploy to S3') {
//    when {
//        expression {
//            params.s3_deploy == true
//        }
//    }
//    steps {
//        sh '$HOME/bin/S3_deploy.sh'
//        echo "S3 deploy is complete"
//    }
//}
}




post {

    success {
        notifySlack("", slackNotificationChannel, [
                [
                        "title": "${env.JOB_NAME}, build #${env.BUILD_NUMBER}",
                        "title_link": "${env.BUILD_URL}",
                        "color": "#00FF00",
                        "text": "Build Success!",
                        "fields": [
                                [
                                        "title": "Branch",
                                        "value": "${env.GIT_BRANCH}",
                                        "short": true
                                ]
                        ]
                ]
        ])

    }
    failure {
        notifySlack("", slackNotificationChannel, [
                [
                        "title": "${env.JOB_NAME}, build #${env.BUILD_NUMBER}",
                        "title_link": "${env.BUILD_URL}",
                        "color": "#FF0000",
                        "text": "Build Failure!",
                        "fields": [
                                [
                                        "title": "Branch",
                                        "value": "${env.GIT_BRANCH}",
                                        "short": true
                                ]
                        ]
                ]
        ])

    }
    unstable {
        notifySlack("", slackNotificationChannel, [
                [
                        "title": "${env.JOB_NAME}, build #${env.BUILD_NUMBER}",
                        "title_link": "${env.BUILD_URL}",
                        "color": "#FF0000",
                        "text": "Build Unstable!",
                        "fields": [
                                [
                                        "title": "Branch",
                                        "value": "${env.GIT_BRANCH}",
                                        "short": true
                                ]
                        ]
                ]
        ])

    }

}
}
