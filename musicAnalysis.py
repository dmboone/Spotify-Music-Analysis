# -*- coding: utf-8 -*-
"""
Spotify Music Analysis

Created by Destiny Boone

Date: April 2020

Description: A program that analyzes your playlists and makes predictions
about what songs you would like and dislike
"""

import spotipy
import spotipy.util as util
import matplotlib.pyplot as plt
import numpy as np
import random
import math

scope = 'playlist-read-private'
username = input("Spotify username: ")
print()

"""Enter values for client_id and client_secret"""       
token = util.prompt_for_user_token(username, 
                                   scope, 
                                   client_id='', 
                                   client_secret='', 
                                   redirect_uri='http://localhost/')

"""Enter values for likedSongsID and dislikedSongsID"""  
isAuthentic = False
likedSongsID = ''
dislikedSongsID = ''
liked = -1
disliked = -1
songs = []
numFeatures = 13  
numTrain = 360
numTest = 240
numLiked = 300
numDisliked = 300
testX = np.zeros([numTest, numFeatures + 1])
testY = np.zeros([numTest, 1])
trainX = np.zeros([numTrain, numFeatures + 1])
trainY = np.zeros([numTrain, 1])
weightsW = np.zeros([numFeatures+1,1])
iterations = 0
alpha = 0.1
costJ = 0
initialJ = 0
finalJ = 0
TP = 0
TN = 0
FP = 0
FN = 0
accuracy = 0
precision = 0
recall = 0
F1 = 0

"""Spotify Verification""" 
if token:
    sp = spotipy.Spotify(auth=token)
    isAuthentic = True;
    results = sp.user_playlist_tracks(username,likedSongsID)
    liked = results["items"]
    while results['next']:
        results = sp.next(results)
        liked.extend(results['items'])
        
    results = sp.user_playlist_tracks(username,dislikedSongsID)
    disliked = results["items"]
    while results['next']:
        results = sp.next(results)
        disliked.extend(results['items'])
    
    songs = liked + disliked
    """shufSongs = random.sample(songs, len(songs))"""
else:
    print("Can't get token for")

"""Creates table from extracted music data"""
if isAuthentic:
    totalSet = np.zeros([len(songs),numFeatures + 1])
    likedSet = np.zeros([numLiked,numFeatures + 1])
    dislikedSet = np.zeros([numDisliked,numFeatures + 1])
    row = 0
    for song in songs:
        title = song["track"]["name"]
        songID = song["track"]["id"]
        analysis = sp.audio_features(songID)
        if song in liked:
            totalSet[row][0] = 1
            likedSet[row][0] = 1
        else:
            totalSet[row][0] = 0
            dislikedSet[row - numLiked][0] = 0
 
        totalSet[row][1] = analysis[0]['acousticness']
        totalSet[row][2] = analysis[0]["danceability"]
        totalSet[row][3] = analysis[0]["duration_ms"] * .000001
        totalSet[row][4] = analysis[0]["energy"]
        totalSet[row][5] = analysis[0]["instrumentalness"]
        totalSet[row][6] = analysis[0]["key"] 
        totalSet[row][7] = analysis[0]["liveness"]
        totalSet[row][8] = analysis[0]["loudness"]
        totalSet[row][9] = analysis[0]["mode"]
        totalSet[row][10] = analysis[0]["speechiness"]
        totalSet[row][11] = analysis[0]["tempo"] / 360
        totalSet[row][12] = analysis[0]["time_signature"]
        totalSet[row][13] = analysis[0]["valence"]
        
        if row < numLiked:
            for i in range(len(totalSet[row]) - 1):
                likedSet[row][i+1] = totalSet[row][i+1]
        else:
            for i in range(len(totalSet[row]) - 1):
                dislikedSet[row - numLiked][i+1] = totalSet[row][i+1]
        row += 1
        
    np.random.shuffle(totalSet)
    for j in range(len(totalSet)):  
        if j < numTrain:
            trainX[j][0] = 1
            trainY[j][0] = totalSet[j][0]
            for i in range(len(totalSet[j]) - 1):
                trainX[j][i+1] = totalSet[j][i+1]
        else:
            testX[j - numTrain][0] = 1
            testY[j - numTrain][0] = totalSet[j][0]
            for i in range(len(totalSet[j]) - 1):
                testX[j - numTrain][i+1] = totalSet[j][i+1]

    """Trains Logistic Regression Algorithm using Training Set"""              
    while iterations != 1000:
        sum = 0
        for k in range(numTrain):
            hw = 1 / (1 + math.exp(-1*np.dot(trainX[k], weightsW)))
            sum += (trainY[k,0]*np.log(hw)) + (1-trainY[k,0])*np.log(1 - hw)
            
        costJ = (-1/numTrain)*sum
        
        if iterations == 0:
            initialJ = costJ
            
        plt.scatter(iterations, costJ, color='black', marker='o')
        
        iterations += 1
               
        for k in range(numFeatures+1):
            sum = 0
            for j in range(numTrain):
                hw = 1 / (1 + math.exp(-1*np.dot(trainX[j],weightsW)))
                sum += (hw - trainY[j,0])*trainX[j,k]
            weightsW[k] = weightsW[k] - (alpha*(1 / numTrain)*sum)
        
    plt.xlabel("Iterations")
    plt.ylabel("J")
    plt.savefig("CostJ"+ ".png", bbox_inches="tight")
    plt.show()
    
    """Calculates Final Stats using Testing Set"""
    for k in range(numTest):
        sum = 0
        hw = 1 / (1 + math.exp(-1*np.dot(testX[k], weightsW)))
        sum += (testY[k,0]*np.log(hw)) + (1-testY[k,0])*np.log(1 - hw)
        
        if hw >= 0.5 and testY[k,0] == 1:
            TP += 1
        elif hw >= 0.5 and testY[k,0] == 0:
            FP += 1
        elif hw < 0.5 and testY[k,0] == 1:
            FN += 1
        else:
            TN += 1
        
    finalJ = (-1/numTest)*sum
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (1 / ((1 / precision) + (1 / recall)))
     
    """Prints Stats on Algorithm"""
    print("\nFinal J: " + str(finalJ))
    print("\nFP: " + str(FP))
    print("\nFN: " + str(FN))
    print("\nTP: " + str(TP))
    print("\nTN: " + str(TN))
    print("\nAccuracy: " + str(accuracy))
    print("\nPrecision: " + str(precision))
    print("\nRecall: " + str(recall))
    print("\nF1: " + str(F1))
   
    """Uncomment the following lines for stats on your personal playlist!"""
    """
    likedAcousticness = likedSet[:,1]
    plt.hist(likedAcousticness, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedAcousticness = dislikedSet[:,1]
    plt.hist(dislikedAcousticness, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Acousticness")
    plt.xlabel("Acousticness")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Acousticness"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedDanceability = likedSet[:,2]
    plt.hist(likedDanceability, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedDanceability = dislikedSet[:,2]
    plt.hist(dislikedDanceability, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Danceability")
    plt.xlabel("Danceability")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Danceability"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedDuration = likedSet[:,3]
    plt.hist(likedDuration, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedDuration = dislikedSet[:,3]
    plt.hist(dislikedDuration, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Duration")
    plt.xlabel("Duration (ms)")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Duration"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedEnergy = likedSet[:,4]
    plt.hist(likedEnergy, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedEnergy = dislikedSet[:,4]
    plt.hist(dislikedEnergy, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Energy")
    plt.xlabel("Energy")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Energy"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedInstrumental = likedSet[:,5]
    plt.hist(likedInstrumental, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedInstrumental = dislikedSet[:,5]
    plt.hist(dislikedInstrumental, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Instrumentalness")
    plt.xlabel("Instrumentalness")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Instrumental"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedKey = likedSet[:,6]
    plt.hist(likedKey, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedKey = dislikedSet[:,6]
    plt.hist(dislikedKey, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Key")
    plt.xlabel("Key")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Key"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedLiveness = likedSet[:,7]
    plt.hist(likedLiveness, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedLiveness = dislikedSet[:,7]
    plt.hist(dislikedLiveness, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Liveness")
    plt.xlabel("Liveness")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Liveness"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedLoudness = likedSet[:,8]
    plt.hist(likedLoudness, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedLoudness = dislikedSet[:,8]
    plt.hist(dislikedLoudness, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Loudness")
    plt.xlabel("Loudness")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Loudness"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedMode = likedSet[:,9]
    plt.hist(likedMode, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedMode = dislikedSet[:,9]
    plt.hist(dislikedMode, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Mode")
    plt.xlabel("Mode")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Mode"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedSpeechiness = likedSet[:,10]
    plt.hist(likedSpeechiness, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedSpeechiness = dislikedSet[:,10]
    plt.hist(dislikedSpeechiness, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Speechiness")
    plt.xlabel("Speechiness")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Speechiness"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedTempo = likedSet[:,11]
    plt.hist(likedTempo, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedTempo = dislikedSet[:,11]
    plt.hist(dislikedTempo, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Tempo (Beats per Minute)")
    plt.xlabel("Tempo")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Tempo"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedTimeSig = likedSet[:,12]
    plt.hist(likedTimeSig, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedTimeSig = dislikedSet[:,12]
    plt.hist(dislikedTimeSig, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Time Signature")
    plt.xlabel("Time Signature")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Time_Signature"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    
    likedValence = likedSet[:,13]
    plt.hist(likedValence, color="skyblue", bins=15, label='Liked', alpha=0.5)
    dislikedValence = dislikedSet[:,13]
    plt.hist(dislikedValence, color="orange", bins=15, label='Disliked', alpha=0.5)
    plt.title("Your Music Analysis - Valence")
    plt.xlabel("Valence")
    plt.ylabel("Number of Songs")
    plt.legend()
    plt.savefig("Valence"+ ".png", bbox_inches="tight")
    plt.show()
    plt.clf()
    """

