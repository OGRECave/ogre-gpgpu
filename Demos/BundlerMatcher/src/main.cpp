#include "BundlerMatcher.h"

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		std::cout << "Usage: " << argv[0] << " <list.txt> <outfile matches> <matchThreshold> <firstOctave>" <<std::endl;
		std::cout << "<matchThreshold> : 0.0 means few match and 1.0 many match (float)" <<std::endl;
		std::cout << "<firstOctave>: specify on which octave start sampling (int)" <<std::endl;
		std::cout << "<firstOctave>: low value (0) means many features and high value (2) means less features" << std::endl;		
		std::cout << "Example: " << argv[0] << " list.txt gpu.matches.txt 0.8 1" << std::endl;

		return -1;
	}

	BundlerMatcher matcher((float) atof(argv[3]), atoi(argv[4]));
	matcher.open(std::string(argv[1]), std::string(argv[2]));

	return 0;
}
