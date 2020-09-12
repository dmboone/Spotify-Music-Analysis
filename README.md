# Spotify-Music-Analysis
Analyze and predict your musical preferences using Spotify's API and Logistic Regression

# Set-Up:
1. Make sure the following dependencies are installed in your python environment:
   * Spotipy (https://pypi.org/project/spotipy/)
   * Matplotlib
   * Numpy
    
2. Will need to register your application with spotify and enter your unique client id and client secret into the parameters listed in the token variable in the program. 
   * See instructions here: https://developer.spotify.com/documentation/general/guides/app-settings/#register-your-app

3. Will need to create two playlists in spotify: 
   * 1 playlist with 300 songs you like
   * 1 playlist with 300 songs you dislike
   * If you want to use different quantities, just be sure to change the values for the following variables in the program to something appropriate:
      * numTrain - the number of songs you want to use in your training set
      * numTest - the number of songs you want to use in your test set (total songs - numTrain)
      * numLiked - the number of songs in your liked playlist
      * numDisliked - the number of songs in your disliked playlist

4. Will then need to copy these playlist ids into the variables likedSongsID and dislikedSongsID
   * You can find the playlist ids by navigating to the playlist online and copying the string of characters at the very end of the link, after "/playlist/"
    
# Instructions to Run:
1. The first time you run, you will be prompted for your spotify username, then asked to verify permissions online through Spotify. You can find your Spotify username in your Account Overview

2. Once verified, you will be redirected to a local host site in your browser; copy the entire site link and paste in console when prompted

3. Output: 
   * Chart tracking the number of iterations and the error J
   * Final J value (want to minimize as close to zero as possible)
   * \# of false positives
   * \# of false negatives
   * \# of true positives
   * \# of true negatives
   * Accuracy
   * Precision
   * Recall
   * F1 Value
   
4. If you want to re-run, be sure to delete the cache file that appears in the folder where you run the project 
   * Should be named ".cache-\<spotify_username\>"
    
# Other Notes:
   * See commented out sections to get more detailed data visualizations for various stats on your playlist! PNGs of this data will be saved to your project folder
   * The algorithm runs 1000 times, so results on your data will likely take several minutes of computing time! Expect longer if you are printing out stats on your music as well!
   * Best recorded accuracy : 70%

# Happy Streaming!
