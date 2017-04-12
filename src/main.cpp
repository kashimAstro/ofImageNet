#include "ofMain.h"
#ifdef SHIFT
#undef SHIFT
#endif

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

#include "imageNet.h"

class ofApp : public ofBaseApp
{
	public:
	std::vector<std::string> out;
	std::vector<ofRectangle> rect;
	std::vector<ofImage> crop;

	ofImage img;

	ofImageNet net;

	void setup()
	{
		img.load("test.jpg");

		net.setup(img);
		out  = net.search();
		rect = net.getRectRandomlyCrop();
		crop = net.getImageCrop();
	}

	void update()
	{
	        ofSetWindowTitle(ofToString(ofGetFrameRate()));
	}

	void draw()
	{
		img.draw(0,0);

		ofPushStyle();
		ofNoFill();
		ofSetColor(ofColor::red);
		for (int i = 0; i < rect.size(); i++)
			ofDrawRectangle(rect[i]);
		ofPopStyle();
			
		for (int i = 0; i < crop.size(); i++)
			crop[i].draw( i*(crop[i].getWidth()/2),img.getHeight(), crop[i].getWidth()/2,crop[i].getHeight()/2 );

		for (int i = 0; i < out.size(); i++)
			ofDrawBitmapStringHighlight(out[i],10,20+(i*20),ofColor::red,ofColor::white);
	}
};

int main(int argc, char *argv[])
{
	ofSetupOpenGL(1024,768, OF_WINDOW);
	ofRunApp( new ofApp());
}
