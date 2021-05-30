from logging import debug
from flask import Flask, views, request, render_template, url_for,send_file, send_from_directory, safe_join, abort
import lxml
from werkzeug.utils import secure_filename
import tensorflow as tf
import os 
from model import video_processing

# Initializing the Flask Application 
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = './video'


@app.route('/upload', methods=["GET","POST"])
def api():
    if(request.method=="GET"):
        return render_template('index.html')

    if(request.method=="POST"):
        link=request.form['video_url']
        print(link)
        left_average_density, right_average_density,passing_junction_density,peak_second, average_stop_time=video_processing.vehicle_detection(link)
        f = open("output.mp4", "r")
        return render_template('video.html',left_average_density=left_average_density, right_average_density=right_average_density,passing_junction_density=passing_junction_density,
        peak_second=peak_second, average_stop_time=average_stop_time)

@app.route('/output.mp4', methods=["GET"])
def file(): 
    if(request.method=="GET"):
        return send_file('output.mp4')


if __name__=="__main__":
    app.run(debug=True)