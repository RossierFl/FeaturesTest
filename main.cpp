#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cv.h>
#include <string>
#include <dirent.h>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <list>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"


using namespace std;
using namespace cv;

cv::Mat get_hogdescriptor_visual_image(cv::Mat& origImg,
                                   cv::vector<float>& descriptorValues,
                                   cv::Size winSize,
                                   cv::Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor);
//void extract_hog_feature(Mat& origImg);
void extract_hog_feature(Mat src);
void getImageFromLiveStream(); 
void listeRepertoire(char* rep);
void extractFeatures(string name, string location);
double getSolidity(Mat src, Mat src_mask);
Scalar getMeanColor(Mat src, Mat src_mask);
void getSURFFeatures(Mat src, Mat src_mask);
bool customKeyPointSort(KeyPoint& a, KeyPoint& b);
ofstream fileOut;
//void thresh_callback(int, void* );
RNG rng(12345); 
int thresh = 100;
int max_thresh = 255;
int minHessianSURF = 100;
list<int> list;
Mat src_gray_gray;
CvCapture* cap;
string location;
string out_file = "/svm_light/test_hog/testing.dat";
    

int main(int argc, char** argv)
{
    /*img_raw = cvLoadImage("/home/rossier/Dropbox/Sources/sampleThumb/car2.jpg", CV_LOAD_IMAGE_UNCHANGED); // load as color image
    cv::Mat img_resized;
   // Mat src_grey;
    //resize(img_raw, img_resized, cv::Size(64,128));
     /// Convert image to gray and blur it
    //cvtColor( img_raw, src_grey, CV_BGR2GRAY );
    blur( img_raw, src_grey, Size(10,10) );
    
    blur( img_raw, src_gray_gray, Size(5,5));
    blur( src_gray_gray, src_gray_gray, Size(5,5));*/

    //blur( src_grey, src_grey, Size(5,5) );
    location = argv[1];
    fileOut.open(out_file.c_str(),std::ostream::binary);
    listeRepertoire(argv[1]);
    fileOut.close();
    //getImageFromLiveStream();
    //namedWindow("Output windows");
    //createTrackbar( " Canny thresh:", "Output windows", &thresh, max_thresh, 0 );
    
    cout<<"modifcommit"<<endl;
    waitKey(0);
}

void listeRepertoire(char* rep)
{
   DIR *dir;
   string ext_exclude = ".json";
   string mask_exclude = "_mask";
   struct dirent *ent;
   if ((dir = opendir (rep)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            string name = ent->d_name;
            //if(ent->d_name)
            if ((name.find(ext_exclude) == std::string::npos)&&(name.find(mask_exclude) == std::string::npos)&&(name != "..")&&(name != ".")) {
                extractFeatures(name,location);
                //return;
            }
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("marche pas");
    }

}

void extractFeatures(string name, string location){
    string fileNameWithoutExtension = name.substr(0, name.rfind("."));
    //fileNameWithoutExtension
    name = location+"/"+name;
    string name_mask = location+"/"+fileNameWithoutExtension + "_mask"+".jpg";
    string json_file = location+"/"+fileNameWithoutExtension+".json";
    //cout<<json_file<<endl;
    Mat src = imread(name,CV_LOAD_IMAGE_GRAYSCALE);
    //Mat src_grey = 
    Mat src_mask = imread(name_mask,CV_LOAD_IMAGE_GRAYSCALE);
    
    
    /*Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
   // std::ifstream test("testis.json", std::ifstream::binary);
    std::ifstream fileContent(json_file.c_str(),std::ifstream::binary);
    bool parsingSuccessful = reader.parse( fileContent, root ,false);
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return;
    }
    root["features"]["solidity"] = getSolidity(src.clone(),src_mask.clone());
    string out_file = location+"/"+fileNameWithoutExtension+"_s.json";
    ofstream fileOut(out_file.c_str(),std::ostream::binary);
    fileOut<<root<<endl;*/
    //cout<<getSolidity(src.clone(),src_mask.clone())<<endl;
    //extract_hog_feature(src.clone());
    getSURFFeatures(src.clone(),src_mask.clone());
    //namedWindow("Output windows");
   // imshow("Output windows",src);
    //cout<<getMeanColor(src.clone(),src_mask.clone())<<endl;
}

double getSolidity(Mat src, Mat src_mask){
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point> outConvexHull;
    RNG rng(12345);
    Mat src_mask_grey;
    findContours( src_mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    double maxContourArea = -1;
    double maxCoutourAreaIdx = -1;
    vector<vector<Point> > hull(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        convexHull(contours[i],hull[i],false);
        double tmp = contourArea(contours[i]);
        if(tmp>maxContourArea){
            maxContourArea=tmp;
            maxCoutourAreaIdx = i;
        }
    }
    convexHull(contours[maxCoutourAreaIdx],outConvexHull);
    double hullArea = contourArea(outConvexHull);
    double solidity = maxContourArea/hullArea;
    
    Mat draw = src.clone();
    Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
    //drawContours(draw,contours,maxCoutourAreaIdx,color,2,8);
    color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
    drawContours(draw,hull,maxCoutourAreaIdx,color,2,8);
    imshow("area",draw);
    waitKey(0);
    return solidity;
}

Scalar getMeanColor(Mat src, Mat src_mask){
    return mean(src,src_mask);
}

void getSURFFeatures(Mat src, Mat src_mask){
    SurfFeatureDetector surf(minHessianSURF);
    
    vector<KeyPoint> keyPoint_1;
    threshold(src_mask,src_mask,128,255,THRESH_BINARY|THRESH_OTSU);
    Mat src_masked;
    src.copyTo(src_masked,src_mask);
    
    surf.detect(src,keyPoint_1);
    
    Mat key_point_img = src.clone();
    imshow("origImage",src_masked);
    imshow("src",src);
    imshow("mask",src_mask);
    waitKey(0);
    drawKeypoints(src,keyPoint_1,key_point_img,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    //imshow("key",key_point_img);
    //imshow("key1",src);
    //imshow("key2",src_mask);
    
    for (int i = 0; i < keyPoint_1.size(); i++) {
        KeyPoint point = keyPoint_1[i];
        int x_coord= point.pt.x-5;
        int y_coord= point.pt.y-5;
        Rect rect(x_coord,y_coord,12,12);
        Mat subImage(src,rect);
        //extract_hog_feature(subImage);
        //rectangle(key_point_img,rect,CV_RGB(255,0,0),1,8);
        
    }
    imshow("key",key_point_img);
    waitKey(0);
    //waitKey(0);
}


void getImageFromLiveStream(){
    VideoCapture vcap;
    cap =cvCreateFileCapture("rtsp://160.98.61.2:554/axis-media/media.amp?videocodec=h264");
    
    
    Mat img_prev;
    Mat img_next;
    vector<Point2f> features_prev, feature_next;
    //rtsp://160.98.61.2:554/axis-media/media.amp?videocodec=h264
    const string videoAdresse="rtsp://160.98.61.2:554/axis-media/media.amp?videocodec=h264";
    if(!vcap.open(videoAdresse)){
        cout<<"Error while opening camera stream"<<endl;
        return;
    }
    namedWindow("Output windows");
    bool first = true;
    while(true){
        
        if(!first)
        {
            img_prev = img_next.clone();
            features_prev = feature_next;
        }
        /*if(!vcap.read(img_next)){
            cout<<"No frame"<<endl;
            continue;
        }*/
        img_next = cvQueryFrame(cap);
        cvtColor( img_next, img_next, CV_BGR2GRAY );
        if(first){ 
            //goodFeaturesToTrack(img_next,feature_next,15,0.01,10);
            first = !first;
        }
        else{
            vector<uchar> status;
            vector<float> err;
            //cout<<features_prev.size();
            //calcOpticalFlowPyrLK(img_prev,img_next,features_prev,feature_next,status,err);
            
        }
        for (int i = 0; i < feature_next.size(); i++) {
            circle(img_next, feature_next[i], 10, Scalar(0,255,0), 0, 8);
        }

        
        namedWindow("Output windows");
        imshow("Output windows",img_next);
        ///thresh_callback( 0, 0 );
        /*namedWindow("Output windowst");
        imshow("Output windowst",img_prev);*/
        if(waitKey(10)>=0)break;
    }
    
}

void extract_hog_feature(Mat src){
        HOGDescriptor d;
    // Size(128,64), //winSize
    // Size(16,16), //blocksize
    // Size(8,8), //blockStride,
    // Size(8,8), //cellSize,
    // 9, //nbins,
    // 0, //derivAper,
    // -1, //winSigma,
    // 0, //histogramNormType,
    // 0.2, //L2HysThresh,
    // 0 //gammal correction,
    // //nlevels=64
    //);
    src.resize(128,64);
     // HOGDescriptor hog(src,Size(16,16),Size(8,8),Size(8,8),8,0,-1,0,0.2,0,64);

    // void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
    //                             Size winStride, Size padding,
    //                             const vector<Point>& locations) const
    vector<float> descriptorsValues;
    vector<Point> locations;
    d.compute(src, descriptorsValues, Size(0,0), Size(0,0), locations);

    //cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
    //cout << "img dimensions: " << src.cols << " width x " << src.rows << "height" << endl;
    //cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
    //cout << "Nr of locations specified : " << locations.size() << endl;
    
    cv::Mat image = get_hogdescriptor_visual_image(src,descriptorsValues,src.size(),cv::Size(8,8),1,1);
    namedWindow( "show HOG", cv::WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "show HOG", image);
    cvWaitKey(0);
    
    /*fileOut<<+1<<" ";
    for(int i=0; i<descriptorsValues.size();i++) {
        fileOut << i+1<<":"<<descriptorsValues.at(i)<<" ";
    }
    fileOut<<endl;    */
    
}


cv::Mat get_hogdescriptor_visual_image(cv::Mat& origImg,
                                   cv::vector<float>& descriptorValues,
                                   cv::Size winSize,
                                   cv::Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor)
{   
    cv::Mat visual_image;
    resize(origImg, visual_image, cv::Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
 
    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14/(float)gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
 
    std::cout << "descriptorDataIdx = " << descriptorDataIdx << std::endl;
 
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            rectangle(visual_image,
                      cv::Point(drawX*scaleFactor,drawY*scaleFactor),
                      cv::Point((drawX+cellSize.width)*scaleFactor,
                      (drawY+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a visual_imagealization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visual_imagealization
                line(visual_image,
                     cv::Point(x1*scaleFactor,y1*scaleFactor),
                     cv::Point(x2*scaleFactor,y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
      for (int x=0; x<cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visual_image;
 
}
