//
//  FLesionImage.cpp
//  featureA
//
//  Created by Ivan Klyuzhin on 2014-11-28.
//  Copyright (c) 2014 isk. All rights reserved.
//

#include "FLesionImage.h"
#include "FModelParams.h"

const double FLesionImage::version = 0.65;

// INITIALIZATION METHODS
FLesionImage::FLesionImage() {
    // TODO Auto-generated constructor stub
}

FLesionImage::~FLesionImage() {
    //the input image must be released externally
    //here release only helper objects
    //Image.release();//only works if image was loaded internally
    //Mask.release();
    //GLCMImage.release();
    //ImageGrayDS.release();
}

FLesionImage::FLesionImage(cv::Mat img)
{
    Image = img;
}

FLesionImage::FLesionImage(std::string path)
{
    Image = cv::imread(path,1);
}

//void FLesionImage::LoadMask(char *path)
//{
//    Mask = cv::imread(path,1);
//}

void FLesionImage::getVersion(double &ver)
{
    ver = version;
}

float FLesionImage::getVersionWeb()
{
    return version;
}


// ---------------------
void FLesionImage::computeGLCM()
{
    //Define parameters for GLCM computation
    
    double numLevels = 32;
    double offsetdata[36] = {-2, 0, 2, 2, -4, 0, 4, 4, -6, 0, 6, 6,
                              2, 2, 2, 0,  4, 4, 4, 0, -6, 0, 6, 6,
                              2, 0, 0, 0,  4, 0, 0, 0, -6, 0, 6, 6}; //emx arrays fill out column-wise
    emxArray_real_T * offsets;
    //emxArray_int32_T * offsets;
    offsets = emxCreateWrapper_real_T(offsetdata, 12, 3);
    boolean_T symmetric = 255;
    
    //Generate grayscale image of type float
    if (ImageGrayDS.empty())
    {
        computeImageGrayDS(Image);
    }

    //Wrap OpenCV into emx and compute the glcm
    double * data = (double*)ImageGrayDS_double.data;
    const emxArray_real_T * tmpmat;
    tmpmat = emxCreateWrapper_real_T(data, ImageGrayDS_double.rows, ImageGrayDS_double.cols);
    
    GLCM_emx = emxCreate_real_T(numLevels, numLevels); //new array
    grayLevelMatrix(tmpmat, numLevels, offsets, symmetric, MaskDS_emx, GLCM_emx);
    
    //Convert matlab array to openCV image
    double * tmpdata = GLCM_emx->data;
    GLCM = cv::Mat(numLevels, numLevels, CV_64FC1, tmpdata);
    //std::cout << "E = " << std::endl << " " << GLCM << std::endl << std::endl;
}

void FLesionImage::computeImageGrayDS(cv::Mat img)
{
    //downsample image and convert to grayscale
    cv::cvtColor(Image, ImageGrayDS, CV_BGR2GRAY, 1);
    cv::resize(ImageGrayDS, ImageGrayDS, cv::Size2i(128, 128), 0, 0, CV_INTER_NN);
    cv::medianBlur(ImageGrayDS, ImageGrayDS, 3);
    
    ImageGrayDS_double = ImageGrayDS.clone();
    ImageGrayDS_double.convertTo(ImageGrayDS_double, CV_64F); //converts to double
    
    double * data = (double*)ImageGrayDS_double.data;
    ImageGrayDS_emx = emxCreateWrapper_real_T(data, ImageGrayDS_double.rows, ImageGrayDS_double.cols);
    computeMaskDS();
}

void FLesionImage::computeMaskDS()
{
    MaskDS = cv::Mat::zeros(128, 128, CV_8U);
    cv::circle(MaskDS, cv::Point2i(64,64), 62, cv::Scalar(255),-1);
    MaskDS = MaskDS > 0;
    boolean_T * data = (boolean_T*)MaskDS.data;
    //ImageGrayDS_emx = emxCreateWrapper_real_T(data, ImageGrayDS_double.rows, ImageGrayDS_double.cols);
    MaskDS_emx = emxCreateWrapper_boolean_T(data, 128, 128);
}

// FEATURE COMPUTATION

//---------------------------------------------ALL FEATURES

std::string FLesionImage::getFeatureString() {
    //compute all features and format them into string
    //char * featValueString[100+];
    char buff[10];
    std::string featValueString;
    std::string hueHistVals;
    std::string grayHistVals;
    std::string featValues;
    std::string allValues;
    //char featValueCString[256];
    char featName[256];
    double featVal = 0;
    double featVer = 0;
    
    //if (grayHistogram.empty()) {
    //    computeGrayHistogram();
    //}
    if (hueHistogram.empty()) {
        computeHueHistogram();
    }
    //Features
    //Hue Bins
    for (int n = 0; n < 16; n++) {
        //get feature name
        float binValue = hueHistogram.at<float>(n,0);
        //add to the value string
        //TO-DO write model parameters as arrays
        switch (n) {
            case 0:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh1, CBIR_FS_STD_hh1);
                break;
            case 1:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh2, CBIR_FS_STD_hh2);
                break;
            case 2:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh3, CBIR_FS_STD_hh3);
                break;
            case 3:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh4, CBIR_FS_STD_hh4);
                break;
            case 4:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh5, CBIR_FS_STD_hh5);
                break;
            case 5:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh6, CBIR_FS_STD_hh6);
                break;
            case 6:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh7, CBIR_FS_STD_hh7);
                break;
            case 7:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh8, CBIR_FS_STD_hh8);
                break;
            case 8:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh9, CBIR_FS_STD_hh9);
                break;
            case 9:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh10, CBIR_FS_STD_hh10);
                break;
            case 10:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh11, CBIR_FS_STD_hh11);
                break;
            case 11:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh12, CBIR_FS_STD_hh12);
                break;
            case 12:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh13, CBIR_FS_STD_hh13);
                break;
            case 13:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh14, CBIR_FS_STD_hh14);
                break;
            case 14:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh15, CBIR_FS_STD_hh15);
                break;
            case 15:
                binValue = normalizeFeature(binValue, CBIR_FS_MIN_hh16, CBIR_FS_STD_hh16);
                break;
            default:
                break;
        }
        sprintf(buff, "%8.4f",binValue);
        hueHistVals += buff;
        hueHistVals += ',';
    }
    int featureNumbers[8] = {0,1,2,3,4,5,6,7};
    //Histogram and wcnn features
    for (int n = 0; n < 8; n++) {
        //get feature name
        int nn = featureNumbers[n];
        this->F_list(nn, featName, featVer);
        //get feature value
        this->F_byname(featName, featVal, featVer);
        //add to the value string
        sprintf(buff, "%8.4f",featVal);
        featValues += buff;
        featValues += ',';
    }
    //NGTD features
    double ngtd[5];
    F_ngtd_all(ngtd, featVer);
    for (int n = 0; n < 5; n++) {
        sprintf(buff, "%8.4f",ngtd[n]);
        featValues += buff;
        featValues += ',';
    }
    //GRAD features
    double grad[3];
    F_grad_all(grad, featVer);
    for (int n = 0; n < 3; n++) {
        sprintf(buff, "%8.4f",grad[n]);
        featValues += buff;
        featValues += ',';
    }
    //add 22 haralick features
    double hhv[22];
    F_glcm_harall(hhv, featVer);
    for (int n = 0; n < 22; n++) {
        sprintf(buff, "%8.4f",hhv[n]);
        featValues += buff;
        featValues += ',';
    }
    //add fractal features
    double frac[1];
    F_frac_all(frac[0], featVer);
    for (int n = 0; n < 1; n++) {
        sprintf(buff, "%8.4f",frac[n]);
        featValues += buff;
        featValues += ',';
    }
    //hu features
    double humi[4];
    F_humi_all(humi, featVer);
    for (int n = 0; n < 4; n++) {
        sprintf(buff, "%8.4f",humi[n]);
        featValues += buff;
        featValues += ',';
    }
    //local features
    double locl[5];
    F_locl_all(locl, featVer);
    for (int n = 0; n < 5; n++) {
        sprintf(buff, "%8.4f",locl[n]);
        featValues += buff;
        if (n < 4) {
            featValues += ',';
        }
    }
    //featValueString.copy(featValueCString, featValueString.length());
    allValues = hueHistVals + featValues;
    return allValues;
}

float FLesionImage::getSimilarityFromFeatures(std::string fstring1, std::string fstring2) {
    
    //std::vector<float> fdata1;
    //std::vector<float> fdata2;
    cv::Mat fdata1, fdata2;
    fdata1.convertTo(fdata1, CV_32F);
    fdata2.convertTo(fdata2, CV_32F);
    //char * featValueString[100+];
    std::string parsedS;
    
    float buff;
    //first string
    std::string inputS = fstring1;
    std::stringstream strStream(inputS);
    while (getline(strStream,parsedS,','))
    {
        buff = stof(parsedS); // do some processing.
        fdata1.push_back(buff);
    }

    //second string
    inputS = fstring2;
    std::stringstream strStream2(inputS);
    while (getline(strStream2,parsedS,','))
    {
        buff = stof(parsedS); // do some processing.
        fdata2.push_back(buff);
    }
    
    //compute grayscale histogram similarity
    //values 1-64
    
    //compute color histogram similarity
    //values 65-128
    
//    //compute grayscale histogram similarity
//    //values 1-64
//    //m(cv::Range(i1,i2),Range(j1,j2))
//    cv::Mat grayHist1 = fdata1(cv::Range(0,64),cv::Range(0,1));
//    cv::Mat grayHist2 = fdata2(cv::Range(0,64),cv::Range(0,1));
//    double grayHistSim = cv::compareHist(grayHist1, grayHist2, CV_COMP_BHATTACHARYYA);
//    //compute color histogram similarity
//    //values 65-128
//    cv::Mat hueHist1 = fdata1(cv::Range(64,128),cv::Range(0,1));
//    cv::Mat hueHist2 = fdata2(cv::Range(64,128),cv::Range(0,1));
//    double hueHistSim = cv::compareHist(hueHist1, hueHist2, CV_COMP_BHATTACHARYYA);
    
    //compute feature similarity
    cv::Mat fVals1 = fdata1(cv::Range(0,64),cv::Range(0,1));
    cv::Mat fVals2 = fdata2(cv::Range(0,64),cv::Range(0,1));
    
    //cv::normalize(fVals1, fVals1, 1, 0, cv::NORM_L2);
    //cv::normalize(fVals2, fVals2, 1, 0, cv::NORM_L2);

//    double ab = fVals1.dot(fVals2);
//    double aa = fVals1.dot(fVals1);
//    double bb = fVals2.dot(fVals2);
//    double cosineSimilarity = ab / sqrt(aa*bb);
//    double harSim = (cosineSimilarity + 1)/2;
    
    cv::Mat dstvec = fVals1 - fVals2;
    double dist = cv::norm(dstvec, cv::NORM_L2);
    return dist;
}

std::string FLesionImage::getHashIndexesForFeatureString(std::string featureString) {
    //compute hash indexes from feature string
    cv::Mat fdata;
    fdata.convertTo(fdata, CV_32F);

    std::string parsedS;
    float buff;

    std::string inputS = featureString;
    std::stringstream strStream(inputS);
    while (getline(strStream,parsedS,','))
    {
        buff = stof(parsedS);
        fdata.push_back(buff);
    }
    cv::Mat fVals = fdata(cv::Range(0,64),cv::Range(0,1));
    //shift the feature values to center around (0.5,0.5, ... 0.5)
    fVals = fVals - 0.5;
    
    int Nsets = 4;
    int Nfeatures = 64;
    int Nplanes = 8;
    
    cv::Mat hashMat = cv::Mat::zeros(Nplanes, Nsets, CV_8U);

    float dotproduct = 0;
    for (int ns = 0; ns < Nsets; ns++) {
        for (int np = 0; np < Nplanes; np++) {
            dotproduct = 0;
            for (int nf = 0; nf < Nfeatures; nf++) {
                dotproduct = dotproduct + (CBIR_LSHvec[ns][np][nf] * fVals.at<float>(nf,0));
            }
            if (dotproduct > 0) {
                hashMat.at<uchar>(np,ns) = 1;
            }
            else {
                hashMat.at<uchar>(np,ns) = 0;
            }
        }
    }
    //write the output string
    std::string binarystr;
    std::string output;
    char buffchar[10];
    char buffind[10];
    int parsed;
    char * ptr;
    
    for (int ns = 0; ns < Nsets; ns++) {
        binarystr.clear();
        for (int np = 0; np < Nplanes; np++) {
            sprintf(buffchar, "%u", hashMat.at<uchar>(np,ns));
            binarystr += buffchar;
        }
        parsed = int(strtol(binarystr.c_str(), & ptr, 2));
        sprintf(buffind, "%u",parsed);
        output += buffind;
        if (ns < Nsets - 1) {
            output += ',';
        }
    }
    return output;
}

void FLesionImage::computeGrayHistogram() {
    //compute gray level hisogram
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    int histSize = 64;
    float range[] = {0,256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::calcHist(&ImageGrayDS, 1, 0, MaskDS, grayHistogram, 1, &histSize, &histRange, uniform, accumulate);
}

void FLesionImage::computeHueHistogram() {
    //compute gray level hisogram
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    cv::Mat ImageHueDS;
    cv::Mat ImageDS;
    cv::resize(Image, ImageDS, cv::Size2i(128,128), 0, 0, CV_INTER_NN);
    cv::cvtColor(ImageDS, ImageDS, CV_BGR2HSV);
    cv::Mat channels[3];
    cv::split(ImageDS, channels);
    ImageHueDS = channels[0];
    
//    cv::Mat tempHistogram;
//    //compute exact histogram
//    int histSize = 256;
//    float range[] = {0,256};
//    const float* histRange = {range};
//    bool uniform = true;
//    bool accumulate = false;
//    cv::calcHist(&ImageHueDS, 1, 0, MaskDS, tempHistogram, 1, &histSize, &histRange, uniform, accumulate);
//    //std::cout << tempHistogram;
//    //filter (smooth) histogram
//    cv::blur(tempHistogram, tempHistogram, cv::Size2i(1,5));
//    //std::cout << tempHistogram;
//    //sum values over several bins
//    hueHistogram = cv::Mat::zeros(16, 1, CV_32FC1);
////    int indRange[] = {0, 8, 16, 24, 32, 40, 48, 56, 64, 114, 148, 180, 212, 223, 234, 245, 256};
//
//    for (int nn = 0; nn < 16; nn++) {
//        int lb = indRange[nn];
//        int ub = indRange[nn+1];
//        double binsum = 0;
//        for (int jj = lb; jj < ub; jj++) {
//            binsum += tempHistogram.data[jj];
//        }
//        hueHistogram.at<float>(nn,0) = binsum;
//        //std::cout << hueHistogram << std::endl;
//    }
//
//    hhVals = [];
//    edges = ;
//    for nn = 1:numel(edges)-1
//        le = edges(nn);
//    ue = edges(nn+1);
//    hhVals(nn,1) = sum(hueHist2(le+1:ue));
//    end
//    hhVals = hhVals/sum(hhVals);
//    
//
    int histSize = 16;
    float range[] = {0,8,16,24,32,40,48,56,64,115,147,179,212,222,233,244,255};
    const float* histRange = {range};
    bool uniform = false;
    bool accumulate = false;
    cv::calcHist(&ImageHueDS, 1, 0, MaskDS, hueHistogram, 1, &histSize, &histRange, uniform, accumulate);
    cv::blur(hueHistogram, hueHistogram, cv::Size2i(1,3), cv::Point2i(-1,-1),cv::BORDER_CONSTANT);
    cv::normalize(hueHistogram, hueHistogram,1,0,cv::NORM_L1);
}

void FLesionImage::F_byname (char * featname, double &val, double &ver) {
    
    if(!strcmp(featname, "hist_vold5090")) F_hist_vold5090(val, ver);
    if(!strcmp(featname, "hist_vald5090")) F_hist_vald5090(val, ver);
    if(!strcmp(featname, "hist_skewness")) F_hist_skewness(val, ver);
    if(!strcmp(featname, "hist_kurtosis")) F_hist_kurtosis(val, ver);
    if(!strcmp(featname, "hist_entropy")) F_hist_entropy(val, ver);
    if(!strcmp(featname, "hist_energy")) F_hist_energy(val, ver);
    
    if(!strcmp(featname, "conn_wcmp")) F_conn_wcmp(val, ver);
    if(!strcmp(featname, "conn_wncc")) F_conn_wncc(val, ver);

}

void FLesionImage::F_list (int featNumber, char * featname, double &ver) {
    switch (featNumber) {
        case 0:
            strcpy(featname, "hist_energy");
            ver = 1;
            break;
        case 1:
            strcpy(featname, "hist_entropy");
            ver = 1;
            break;
        case 2:
            strcpy(featname, "hist_kurtosis");
            ver = 1;
            break;
        case 3:
            strcpy(featname, "hist_skewness");
            ver = 1;
            break;
        case 4:
            strcpy(featname, "hist_vald5090");
            ver = 1;
            break;
        case 5:
            strcpy(featname, "hist_vold5090");
            ver = 1;
            break;
        case 6:
            strcpy(featname, "conn_wcmp");
            ver = 1;
            break;
        case 7:
            strcpy(featname, "conn_wncc");
            ver = 1;
            break;

        default:
            strcpy(featname, "");
            ver = 0;
            break;
    }
}

void FLesionImage::F_glcm_harall(double *val, double &ver) {
    if (GLCM.empty()) {
        computeGLCM();
    }
    f_harall(GLCM_emx, val); //mask is used during the GLCM computation
    val[0] = normalizeFeature(val[0], CBIR_FS_MIN_glcm_autoc, CBIR_FS_MAX_glcm_autoc);
    val[1] = normalizeFeature(val[1], CBIR_FS_MIN_glcm_contr, CBIR_FS_MAX_glcm_contr);
    val[2] = normalizeFeature(val[2], CBIR_FS_MIN_glcm_corrm, CBIR_FS_MAX_glcm_corrm);
    val[3] = normalizeFeature(val[3], CBIR_FS_MIN_glcm_corrp, CBIR_FS_MAX_glcm_corrp);
    val[4] = normalizeFeature(val[4], CBIR_FS_MIN_glcm_cprom, CBIR_FS_MAX_glcm_cprom);
    
    val[5] = normalizeFeature(val[5], CBIR_FS_MIN_glcm_cshad, CBIR_FS_MAX_glcm_cshad);
    val[6] = normalizeFeature(val[6], CBIR_FS_MIN_glcm_dissi, CBIR_FS_MAX_glcm_dissi);
    val[7] = normalizeFeature(val[7], CBIR_FS_MIN_glcm_energ, CBIR_FS_MAX_glcm_energ);
    val[8] = normalizeFeature(val[8], CBIR_FS_MIN_glcm_entro, CBIR_FS_MAX_glcm_entro);
    val[9] = normalizeFeature(val[9], CBIR_FS_MIN_glcm_homom, CBIR_FS_MAX_glcm_homom);
    
    val[10] = normalizeFeature(val[10], CBIR_FS_MIN_glcm_homop, CBIR_FS_MAX_glcm_homop);
    val[11] = normalizeFeature(val[11], CBIR_FS_MIN_glcm_maxpr, CBIR_FS_MAX_glcm_maxpr);
    val[12] = normalizeFeature(val[12], CBIR_FS_MIN_glcm_sosvh, CBIR_FS_MAX_glcm_sosvh);
    val[13] = normalizeFeature(val[13], CBIR_FS_MIN_glcm_savgh, CBIR_FS_MAX_glcm_savgh);
    val[14] = normalizeFeature(val[14], CBIR_FS_MIN_glcm_svarh, CBIR_FS_MAX_glcm_svarh);
    
    val[15] = normalizeFeature(val[15], CBIR_FS_MIN_glcm_senth, CBIR_FS_MAX_glcm_senth);
    val[16] = normalizeFeature(val[16], CBIR_FS_MIN_glcm_dvarh, CBIR_FS_MAX_glcm_dvarh);
    val[17] = normalizeFeature(val[17], CBIR_FS_MIN_glcm_denth, CBIR_FS_MAX_glcm_denth);
    val[18] = normalizeFeature(val[18], CBIR_FS_MIN_glcm_inf1h, CBIR_FS_MAX_glcm_inf1h);
    val[19] = normalizeFeature(val[19], CBIR_FS_MIN_glcm_inf2h, CBIR_FS_MAX_glcm_inf2h);
    
    val[20] = normalizeFeature(val[20], CBIR_FS_MIN_glcm_indnc, CBIR_FS_MAX_glcm_indnc);
    val[21] = normalizeFeature(val[21], CBIR_FS_MIN_glcm_idmnc, CBIR_FS_MAX_glcm_idmnc);
    
    ver = version;
}

void FLesionImage::F_ngtd_all(double *val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    f_ngtd_all(ImageGrayDS_emx, MaskDS_emx, val);
    val[0] = normalizeFeature(val[0], CBIR_FS_MIN_ngtd_coars, CBIR_FS_MAX_ngtd_coars);
    val[1] = normalizeFeature(val[1], CBIR_FS_MIN_ngtd_contr, CBIR_FS_MAX_ngtd_contr);
    val[2] = normalizeFeature(val[2], CBIR_FS_MIN_ngtd_busyn, CBIR_FS_MAX_ngtd_busyn);
    val[3] = normalizeFeature(val[3], CBIR_FS_MIN_ngtd_compl, CBIR_FS_MAX_ngtd_compl);
    val[4] = normalizeFeature(val[4], CBIR_FS_MIN_ngtd_stren, CBIR_FS_MAX_ngtd_stren);
    ver = version;
}

void FLesionImage::F_grad_all(double *val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    f_grad_all(ImageGrayDS_emx, MaskDS_emx, val);
    val[0] = normalizeFeature(val[0], CBIR_FS_MIN_gsmi_gradmea, CBIR_FS_MAX_gsmi_gradmea);
    val[1] = normalizeFeature(val[1], CBIR_FS_MIN_gsmi_gradvar, CBIR_FS_MAX_gsmi_gradvar);
    val[2] = normalizeFeature(val[2], CBIR_FS_MIN_gsmi_gradani, CBIR_FS_MAX_gsmi_gradani);
    ver = version;
}

// --------------------------------------------HIST group
void FLesionImage::F_hist_energy(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_histenergy(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_hist_energy, CBIR_FS_MAX_hist_energy);
    ver = version;
}

void FLesionImage::F_hist_entropy(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_histentropy(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_hist_entropy, CBIR_FS_MAX_hist_entropy);
    ver = version;
}

void FLesionImage::F_hist_kurtosis(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_kurtosis(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_hist_kurtosis, CBIR_FS_MAX_hist_kurtosis);
    ver = version;
}

void FLesionImage::F_hist_skewness(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_skewness(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_hist_skewness, CBIR_FS_MAX_hist_skewness);
    ver = version;
}
void FLesionImage::F_hist_vald5090(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_vald5090(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_hist_vald5090, CBIR_FS_MAX_hist_vald5090);
    ver = version;
}
void FLesionImage::F_hist_vold5090(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_vold5090(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_hist_vold5090, CBIR_FS_MAX_hist_vold5090);
    ver = version;
}

//--------------------------------------------CONN group
void FLesionImage::F_conn_wcmp(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_wcmp(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_conn_wcmp, CBIR_FS_MAX_conn_wcmp);
    ver = version;
}

void FLesionImage::F_conn_wncc(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_wncc(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_conn_wncc, CBIR_FS_MAX_conn_wncc);
    ver = version;
}

//--------------------------------------------FRAC group
void FLesionImage::F_frac_all(double &val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    val = f_frac_all(ImageGrayDS_emx, MaskDS_emx);
    val = normalizeFeature(val, CBIR_FS_MIN_frac_fd, CBIR_FS_MAX_frac_fd);
    ver = version;
}
//--------------------------------------------HUMI group
void FLesionImage::F_humi_all(double *val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    f_humi_all(ImageGrayDS_emx, MaskDS_emx, val);
    val[0] = normalizeFeature(val[0], CBIR_FS_MIN_humi_j1, CBIR_FS_MAX_humi_j1);
    val[1] = normalizeFeature(val[1], CBIR_FS_MIN_humi_j2, CBIR_FS_MAX_humi_j2);
    val[2] = normalizeFeature(val[2], CBIR_FS_MIN_humi_b3, CBIR_FS_MAX_humi_b3);
    val[3] = normalizeFeature(val[3], CBIR_FS_MIN_humi_b4, CBIR_FS_MAX_humi_b4);
    ver = version;
}
//--------------------------------------------LOCL group
void FLesionImage::F_locl_all(double *val, double &ver) {
    if (ImageGrayDS.empty()) {
        computeImageGrayDS(Image);
    }
    f_locl_all(ImageGrayDS_emx, MaskDS_emx, val);
    val[0] = normalizeFeature(val[0], CBIR_FS_MIN_locl_sum, CBIR_FS_MAX_locl_sum);
    val[1] = normalizeFeature(val[1], CBIR_FS_MIN_locl_std, CBIR_FS_MAX_locl_std);
    val[2] = normalizeFeature(val[2], CBIR_FS_MIN_locl_skew, CBIR_FS_MAX_locl_skew);
    val[3] = normalizeFeature(val[3], CBIR_FS_MIN_locl_kurt, CBIR_FS_MAX_locl_kurt);
    val[4] = normalizeFeature(val[4], CBIR_FS_MIN_locl_fwhm, CBIR_FS_MAX_locl_fwhm);
    ver = version;
}

//--------
double FLesionImage::normalizeFeature(double value, double valuemin, double valuemax) {
    //normalize the feature value to zero mean and standard deviation = 1;
    double amp = valuemax - valuemin;
    double newvalue;
    if (amp != 0) {
         newvalue = (value - valuemin)/amp;
    }
    else
        newvalue = value;
    return newvalue;
}