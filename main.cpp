
#include <opencv2/core/core.hpp> 
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <iostream> 

IplImage* image = 0;
IplImage* dst = 0;
int vals = 0;


void findRectangles(IplImage* _image) 
{ 
assert(_image!=0); 
 
IplImage* bin = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1); 
// ������������ � �������� ������ 
cvConvertImage(_image, bin, CV_BGR2GRAY); 
// ������� ������� 
cvCanny(bin, bin, 50, 200); 
cvNamedWindow( "bin", 1 ); 
cvShowImage("bin", bin); 
// ��������� ������ ��� �������� 
CvMemStorage* storage = cvCreateMemStorage(0); 
CvSeq* contours=0; 
 
// ������� ������� 
int contoursCont = cvFindContours( bin, storage,&contours,sizeof(CvContour),CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE,cvPoint(0,0)); 
 
assert(contours!=0); 

// ������� ��� ������� 
for( CvSeq* current = contours; current != NULL; current = current->h_next ){ 
// ��������� ������� � �������� ������� 
double area = fabs(cvContourArea(current)); 
double perim = cvContourPerimeter(current); 
 
 
if ( area / (perim*perim/16) > 0.7 && area / (perim*perim/16) < 1.3 ){ // � 10% ��������� 
	vals++;
cvDrawContours(_image, current, cvScalar(0, 0, 255), cvScalar(0, 255, 0), -1, 1, 8); 
} 
} 
 
 
cvReleaseMemStorage(&storage); 
cvReleaseImage(&bin); 
} 


int main(int argc, char* argv[])
{
        char* filename = argc == 2 ? argv[1] : "Sqrs2.jpg";
        image = cvLoadImage(filename,1); 
        dst = cvCloneImage(image);
      //  printf("[i] image: %s\n", filename);
        assert( image != 0 );
        // ���� ��� ����������� ��������
       // cvNamedWindow("original",CV_WINDOW_AUTOSIZE);
     // cvNamedWindow("smooth",CV_WINDOW_AUTOSIZE);
        // ���������� �������� ��������
        //cvSmooth(image, dst, CV_GAUSSIAN, 3, 3);
		//cvSmooth(image, dst, CV_GAUSSIAN, 3, 3);

		//cvSmooth(image, dst, CV_BILATERAL, 3, 3);
		//cvSmooth(image, dst, CV_GAUSSIAN, 3, 3);
		cvSmooth(image, dst, CV_MEDIAN ,3, 3 );

			cvNamedWindow( "final", 1 ); 
			cvShowImage( "final", dst); 
			//dst = cvCloneImage(image); 
			// ������� ����� �� ����������� 
			findRectangles(dst); 
			cvNamedWindow( "rectangles", 1 ); 
			cvShowImage( "rectangles", dst); 
			// ��� ������� ������� 
			cvWaitKey(0);  
			// ����������� ������� 
			cvReleaseImage(&dst); 
			cvReleaseImage(&dst); 
			// ������� ���� 
			cvDestroyAllWindows(); 
			return 0; 
}