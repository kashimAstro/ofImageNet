#include "ofMain.h"
#ifdef SHIFT
#undef SHIFT
#endif

using namespace dlib;
using namespace std;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using level1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = ares<64,ares<64,ares<64,SUBNET>>>;

using anet_type = loss_multiclass_log<fc<1000,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;

class ofImageNet {
	public:

	std::vector<string> labels;
	std::vector<ofRectangle> rect_pred;
	std::vector<ofImage> image_pred;
	anet_type net;
	matrix<rgb_pixel> img, crop;

	rectangle make_random_cropping_rect_resnet(
	    const matrix<rgb_pixel>& img,
	    dlib::rand& rnd
	)
	{
	    double mins = 0.466666666, maxs = 0.875;
	    auto scale = mins + rnd.get_random_double()*(maxs-mins);
	    auto size = scale*std::min(img.nr(), img.nc());
	    rectangle rect(size, size);
	    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
			 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
	    return move_rect(rect, offset);
	}

	void randomly_crop_images (
	    const matrix<rgb_pixel>& img,
	    dlib::array<matrix<rgb_pixel>>& crops,
	    dlib::rand& rnd,
	    long num_crops
	)
	{
	    std::vector<chip_details> dets;
	    for (long i = 0; i < num_crops; ++i)
	    {
		auto rect = make_random_cropping_rect_resnet(img, rnd);
		rect_pred.push_back(ofRectangle(rect.left(),rect.top(),rect.width(),rect.height()));
		dets.push_back(chip_details(rect, chip_dims(227,227)));
	    }

	    extract_image_chips(img, dets, crops);

	    for (auto&& img : crops)
	    {
		if (rnd.get_random_double() > 0.5) {
		    img = fliplr(img);
		    image_pred.push_back(toOf(img));
		}
		apply_random_color_offset(img, rnd);
	    }
	}

	ofPixels toOf(const dlib::matrix<dlib::rgb_pixel> rgb)
        {
            ofPixels p;
            int w = rgb.nc();
            int h = rgb.nr();
            p.allocate(w, h, OF_IMAGE_COLOR);
            for(int y = 0; y<h; y++)
            {
                 for(int x=0; x<w;x++)
                 {
                        p.setColor(x, y, ofColor(rgb(y,x).red,
                                                 rgb(y,x).green,
                                                 rgb(y,x).blue));
                 }
            }
            return p;
        }

        dlib::matrix<dlib::rgb_pixel> toDLib(const ofPixels px)
        {
            dlib::matrix<dlib::rgb_pixel> out;
            int width = px.getWidth();
            int height = px.getHeight();
            int ch = px.getNumChannels();

            out.set_size( height, width );
            const unsigned char* data = px.getData();
            for ( unsigned n = 0; n < height;n++ )
            {
                const unsigned char* v =  &data[n * width *  ch];
                for ( unsigned m = 0; m < width;m++ )
                {
                    if ( ch==1 )
                    {
                        unsigned char p = v[m];
                        dlib::assign_pixel( out(n,m), p );
                    }
                    else{
                        dlib::rgb_pixel p;
                        p.red   = v[m*3];
                        p.green = v[m*3+1];
                        p.blue  = v[m*3+2];
                        dlib::assign_pixel( out(n,m), p );
                    }
                }
            }
            return out;
        }

	void setup(ofImage _ximg, string _resnet = "resnet34_1000_imagenet_classifier.dnn")
	{
		deserialize(ofToDataPath(_resnet)) >> net >> labels;
		img = toDLib(_ximg.getPixels());
	}

	std::vector<std::string> search()
	{
		std::vector<std::string> out;
		softmax<anet_type::subnet_type> snet;
		snet.subnet() = net.subnet();
		dlib::array<matrix<rgb_pixel>> images;
		dlib::rand rnd;


	        const int num_crops = 16;
        	randomly_crop_images(img, images, rnd, num_crops);

        	matrix<float,1,1000> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;
	        for (int k = 0; k < 5; ++k)
        	{
	            unsigned long predicted_label = index_of_max(p);
        	    out.push_back( ofToString(p(predicted_label)) + ": " + ofToString(labels[predicted_label]) );
	            p(predicted_label) = 0;
        	}
		return out;
    	}

	std::vector<ofImage> getImageCrop()
	{
		return image_pred;
	}

	std::vector<ofRectangle> getRectRandomlyCrop()
	{
		return rect_pred;
	}
};
