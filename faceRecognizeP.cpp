#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/face.hpp"
#include<vector>
#include <iostream>
#include<fstream>

using namespace std;
using namespace cv;
using namespace cv::face;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
	        
	ifstream fileIm(filename.c_str(), ifstream::in);
	char separator = ';';
	if (!fileIm) 
	{
		string error_message = "File nonexistent or invalid, verify the path ";
		cout << error_message;
		exit(-1);
	}
	string readLine, path, label, nom;
	while (getline(fileIm, readLine)) {
		stringstream liness(readLine);
		getline(liness, path, separator);
		getline(liness, label);
		if (!path.empty() && !label.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(label.c_str()));

		}

	}
}
static void read_names(const string& filename, vector<string>& names, vector<int>& labels) {

	ifstream file(filename.c_str(), ifstream::in);
	char separator = ';';
	if (!file) {
		string error_message = "Fichier csv invalide ou inexistant. ";
		cout << error_message;
		exit(-1);
	}
	
	string line, name, classlabel, nom;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, name, separator);
		getline(liness, classlabel);
		if (!name.empty() && !classlabel.empty()) {
			names.push_back(name);
			labels.push_back(atoi(classlabel.c_str()));

		}

	}
}
int main(int argc, char** argv) {
	int k = 1;
	string fn_haar = "D:\\op\\opencv-3.4.14\\opencv-3.4.14\\data\\haarcascades\\haarcascade_frontalcatface_extended.xml ";
	string fn_csv = "C:\\Users\\ALAY\\Desktop\\versionfinal\\file1.txt";
	Mat original = imread("C:\\Users\\ALAY\\Desktop\\versionfinal\\imgDB\\im_(201).jpg", IMREAD_COLOR);
	if (original.empty()) {
		cout << "no image";
		exit(0);
	}
	VideoCapture cap(0); 
		Mat gray;
		vector<Mat> images; 
		vector<string> names;
		vector<int> labels, label_name;
		try {
			read_csv(fn_csv, images, labels);
		    }
		catch (Exception& e) {
			cerr << " Erreur Inconnue " << endl;
			exit(1);
		}
		try {
			read_names("..\\names.txt", names, label_name);

		}
		catch (Exception& e) {
			cerr << " Erreur Inconnue " << endl;
			exit(1);

		}
		int im_width = images[0].cols;
		int im_height = images[0].rows;
		Ptr<BasicFaceRecognizer> model = FisherFaceRecognizer::create(10,123.0);
		model->train(images, labels);
		while (1) {
			cap >> original; k++;
		cvtColor(original, gray, COLOR_RGBA2GRAY);
		CascadeClassifier haar_cascade;
		haar_cascade.load(fn_haar);
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);
		if (faces.size() == 0) {
			cout << " Pas de visage detecte " << " \n ";
		}

		else {
			Mat face_resized;
			for (int i = 0; i < faces.size(); i++) {
				Rect face_i = faces[i];
				Mat face = gray(face_i);
				resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				equalizeHist(face_resized, face_resized);
				medianBlur(face_resized, face_resized, 3);
				//imshow("ff", face_resized);
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				int predictionLabel;
				double conf;
				model->predict(face_resized, predictionLabel,conf);

				string box_text, name;
				if (predictionLabel > 0)
				{
					box_text.append(format(" IDentified : %d", predictionLabel));
					for (int j = 0; j < label_name.size(); j++) {
						if (predictionLabel == label_name[j]) name = " " + names[j];
					}

					string result = format("predicted class = %d  / actual class = %d / confidence= %f\n", predictionLabel, labels.size(),conf);
					cout << result;

					imwrite(format("C:/Users/ALAY/Desktop/ima%d.png", labels.size()), face_resized);
				}
				else {
					box_text.append(" Inconnu ");
					resize(face, face, Size(70,100) );
					imwrite(format("C://Users//ALAY//Desktop//versionfinal//ige%d.png", labels.size()), face);
				}

				int pos_x = max((face_i.tl().x - 10), 0);
				int pos_y = max((face_i.br().y + 15), 0);
				putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 0.5, CV_RGB(237, 27, 37), 1.0);
				putText(original, name, Point(pos_x, pos_y + 12), FONT_HERSHEY_PLAIN, 0.5, CV_RGB(237, 27, 37), 1.0);
			}

			/*imshow("face_recognizer", original);

			waitKey(0);*/

		}
		imshow("cap", original);

		waitKey(1);
	}
	return 0;
}
