#include <stdio.h>
#include <stdint.h>

int main(int argc, char *argv[]) {

	FILE *f = fopen("t10k-images-idx3-ubyte", "rb");
	uint8_t b;

	printf("No btyes in an int: %d\n", sizeof(b));

	for(int i = 0; i < 16; i++){
		fread(&b, 1, 1, f);
		printf("%02x", b);
	}

	printf("\n");
	
	for(int k = 0; k < 3; k++){
		for(int i = 0; i < 28; i++){
	 		for(int j = 0; j < 28; j ++){
		 		fread(&b, 1, 1, f);
		 		printf("%s", (b > 127) ? "0" : ".");
	 		}		
	 	printf("\n");
		}
	printf("\n");
	}

	printf("\n");

	return 0;
}
