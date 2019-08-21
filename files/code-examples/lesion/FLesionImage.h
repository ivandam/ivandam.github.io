//
//  FLesionImage.h
//  featureA
//
//  Created by Ivan Klyuzhin on 2014-11-28.
//  Copyright (c) 2014 isk. All rights reserved.
//

#ifndef __featureA__FLesionImage__
#define __featureA__FLesionImage__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
//#include "f_feats_emxAPI.h"
#include "emxfeats.h"
#include "LSHParams.h"

class FLesionImage {
public:
    static double getVersion()
    {
        return version;
    }
    //Initialization
    FLesionImage();
    FLesionImage(cv::Mat img);
    FLesionImage(std::string path);
    float getVersionWeb();
    void LoadImage(char *str);
    void LoadImage(cv::Mat img);
    //void LoadMask(char *str);
    //void LoadMask(cv::Mat mask);
    //Destructor
    ~FLesionImage();
    //Intermediate images
    
    //Feature functions
    //void F_all (double * val, double * ver);
    
    //Compute metrics
    
    //compute all features as a string
    std::string getFeatureString();
    
    //compute similarity for two feature strings
    float getSimilarityFromFeatures(std::string fstring1, std::string fstring2);
    
    //get engine version
    void getVersion (double & version);
    
    //get hash indexes for feature string
    std::string getHashIndexesForFeatureString(std::string);
    
private:
    //Internal storage for image data
    cv::Mat Image;
    cv::Mat Mask;
    cv::Mat MaskDS;
    //cv::Mat ImageOriginal;
    cv::Mat grayHistogram;
    cv::Mat hueHistogram;
    emxArray_real_T * GLCM_emx;
    const emxArray_real_T * ImageGrayDS_emx;
    emxArray_boolean_T * MaskDS_emx;
    
    cv::Mat ImageGrayDS;
    cv::Mat ImageGrayDS_double;
    cv::Mat GLCM;
    
    //Methods
    void computeImageGrayDS(cv::Mat img);
    void computeGLCM();
    void computeMaskDS();
    double normalizeFeature(double value, double mean, double std);
    void computeGrayHistogram();
    void computeHueHistogram();
    
    //Properties
    static const double version;
    int width;
    int height;
    cv::Rect myROI;
    
    //Features
    //return list of features with versions
    void F_list (int featNumber, char * featname, double &ver);
    void F_byname (char * featname, double &val, double &ver);
    
    // --------------------------------------------GLCM group
    void F_glcm_harall (double *val, double &ver);
    
    // --------------------------------------------NGTD group
    void F_ngtd_all (double *var, double &ver);
    
    // --------------------------------------------HIST group
    void F_hist_energy (double &val, double &ver);
    void F_hist_entropy (double &val, double &ver);
    void F_hist_kurtosis (double &val, double &ver);
    void F_hist_skewness (double &val, double &ver);
    void F_hist_vald5090 (double &val, double &ver);
    void F_hist_vold5090 (double &val, double &ver);
    
    // --------------------------------------------GSMI group
    void F_grad_all(double *val, double &ver);
    
    //--------------------------------------------CONN group
    void F_conn_wcmp (double &val, double &ver);
    void F_conn_wncc (double &val, double &ver);
    
    //--------------------------------------------FRAC group
    void F_frac_all (double &val, double &ver);
    //--------------------------------------------HUMI group
    void F_humi_all (double *val, double &ver);
    //--------------------------------------------LOCL group
    void F_locl_all (double *val, double &ver);
};

#endif /* defined(__featureA__FLesionImage__) */
