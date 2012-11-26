/*
   Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 16
#define FALSE   0
#define TRUE    1
#define OFFSET_BITS     12
#define LENGTH_BITS     4
/* We want a sliding window*/
#define WINDOW_SIZE     (1 << OFFSET_BITS)
/* maximum match length not encoded and maximum length encoded (4 bits) */
#define MAX_UNCODED     2
#define MAX_CODED       ((1 << LENGTH_BITS) + MAX_UNCODED)
#define ENCODED         0       /* encoded string */
#define UNCODED         1       /* unencoded character */
#define Wrap(value, limit)      (((value) < (limit)) ? (value) : ((value) - (limit)))

//kernel launch related defines
#define MAX_THREADS_PER_BLOCK   32 
#define MAX_BYTES_PER_BLOCK     24576 //24KB, half of max shared memory per block
#define MAX_BYTES_PER_THREAD    (MAX_BYTES_PER_BLOCK / MAX_THREADS_PER_BLOCK)


const char *outFile = "./myout.txt" ; //Path for writing compressed output buffer to a file

typedef enum
{
    BF_UNKNOWN_ENDIAN,
    BF_LITTLE_ENDIAN,
    BF_BIG_ENDIAN
} endian_t;

struct bit_file_t
{
    unsigned char *outBuffer;
    unsigned int  outBytes;
    unsigned char bitBuffer;    /* bits waiting to be read/written */
    unsigned char bitCount;     /* number of bits in bitBuffer */
    endian_t endian; 
};

/***************************************************************************
 * This data structure stores an encoded string in (offset, length) format.
 * The actual encoded string is stored using OFFSET_BITS for the offset and
 * LENGTH_BITS for the length.
 ***************************************************************************/
typedef struct encoded_string_t
{
    unsigned int offset;    /* offset to start of longest match */
    unsigned int length;    /* length of longest match */
} encoded_string_t;

/* union used to test for endianess */
typedef union
{
    unsigned long word;
    unsigned char bytes[sizeof(unsigned long)];
} endian_test_t;


// cyclic buffer sliding window of already read characters 
__device__ unsigned char *slidingWindow;   //SIZE WINDOW SIZE 
__device__ unsigned char *uncodedLookahead;  //SIZE MAX CODED



/* Prototypes */
__device__ inline static void putInBufferStream(unsigned char c,bit_file_t * stream);
static void printBufferStream(char * buffer,int length);
__device__ inline static int BitFilePutBitsLE(bit_file_t *stream, void *bits, const unsigned int count);
/* get/put character */
__device__ inline static int BitFilePutChar(const int c, bit_file_t *stream);

/* get/put single bit */
__device__ inline static int BitFilePutBit(const int c, bit_file_t *stream);

/* get/put number of bits (most significant bit to least significat bit) */
__device__ inline static int BitFilePutBits(bit_file_t *stream, void *bits, const unsigned int count);
__device__ inline static int BitFilePutBitsInt(bit_file_t *stream, void *bits, const unsigned int count,
        const size_t size);



/**************DEVICE HELPER FUNCTIONS ********************************/ 

/**********MATCHING FUNCTIONS**********************/

/****************************************************************************
 *   Function   : FindMatch
 *   Description: This function will search through the slidingWindow
 *                dictionary for the longest sequence matching the MAX_CODED
 *                long string stored in uncodedLookahed.
 *   Parameters : windowHead - head of sliding window
 *                uncodedHead - head of uncoded lookahead buffer
 *   Effects    : None
 *   Returned   : The sliding window index where the match starts and the
 *                length of the match.  If there is no match a length of
 *                zero will be returned.
 ****************************************************************************/
    __device__ inline static 
encoded_string_t FindMatch(unsigned int windowHead, unsigned int uncodedHead)
{
    encoded_string_t matchData;
    unsigned int i, j;

    matchData.length = 0;
    matchData.offset = 0;
    i = windowHead;  /* start at the beginning of the sliding window */
    j = 0;

    while (TRUE)
    {
        if (slidingWindow[i] == uncodedLookahead[uncodedHead])
        {
            /* we matched one how many more match? */
            j = 1;

            while(slidingWindow[Wrap((i + j), WINDOW_SIZE)] ==
                    uncodedLookahead[Wrap((uncodedHead + j), MAX_CODED)])
            {
                if (j >= MAX_CODED)
                {
                    break;
                }
                j++;
            }

            if (j > matchData.length)
            {
                matchData.length = j;
                matchData.offset = i;
            }
        }

        if (j >= MAX_CODED)
        {
            matchData.length = MAX_CODED;
            break;
        }

        i = Wrap((i + 1), WINDOW_SIZE);
        if (i == windowHead)
        {
            /* we wrapped around */
            break;
        }
    }

    return matchData;
}

/****************************************************************************
 *   Function   : ReplaceChar
 *   Description: This function replaces the character stored in
 *                slidingWindow[charIndex] with the one specified by
 *                replacement.
 *   Parameters : charIndex - sliding window index of the character to be
 *                            removed from the linked list.
 *   Effects    : slidingWindow[charIndex] is replaced by replacement.
 *   Returned   : None
 ****************************************************************************/
    __device__ inline static 
void ReplaceChar(unsigned int charIndex, unsigned char replacement)
{
    slidingWindow[charIndex] = replacement;
}
/**********END OF MATCHING FUNCTIONS**********************/


/**********BITFILE FUNCTIONS**********************/

/* Opening BitStream */
    __device__ inline static 
bit_file_t *BitStreamOpen(unsigned char *buffer)
{
    bit_file_t *bf;

    bf = (bit_file_t *)malloc(sizeof(bit_file_t));

    if (bf == NULL)
    {
        /* malloc failed */
        //	errno = ENOMEM;  Need to set and handle this for CUDA
    }
    else
    {       bf->outBuffer = buffer;
        bf->outBytes = 0;
        bf->bitBuffer = 0;
        bf->bitCount = 0;
        bf->endian = BF_LITTLE_ENDIAN;
    }

    return (bf);
}


/***************************************************************************
 *   Function   : BitFilePutChar
 *   Description: This function writes the byte passed as a parameter to the
 *                file passed a parameter.
 *   Parameters : c - the character to be written
 *                stream - pointer to bit file stream to write to
 *   Effects    : Writes a byte to the file and updates buffer accordingly.
 *   Returned   : On success, the character written, otherwise EOF.
 ***************************************************************************/
    __device__ inline static 
int BitFilePutChar(const int c, bit_file_t *stream)
{
    unsigned char tmp;

    if (stream == NULL)
    {
        return(EOF);
    }

    if (stream->bitCount == 0)
    {

        /* Printing to buffer */
        //putInBuffer(c);
        putInBufferStream(c,stream);
        return c;
    }

    /* figure out what to write */
    tmp = ((unsigned char)c) >> (stream->bitCount);
    tmp = tmp | ((stream->bitBuffer) << (8 - stream->bitCount));


    /* Printing to buffer */
    // putInBuffer(tmp);
    putInBufferStream(tmp,stream);
    /* We shud add stream->bitBuffer = c here */
    stream->bitBuffer = c;  /* VERY CAREFUL */

    return tmp;
}


/***************************************************************************
 *   Function   : BitFilePutBit
 *   Description: This function writes the bit passed as a parameter to the
 *                file passed a parameter.
 *   Parameters : c - the bit value to be written
 *                stream - pointer to bit file stream to write to
 *   Effects    : Writes a bit to the bit buffer.  If the buffer has a byte,
 *                the buffer is written to the file and cleared.
 *   Returned   : On success, the bit value written, otherwise EOF.
 ***************************************************************************/
    __device__ inline static 
int BitFilePutBit(const int c, bit_file_t *stream)
{
    int returnValue = c;

    if (stream == NULL)
    {
        return(EOF);
    }

    stream->bitCount++;
    stream->bitBuffer <<= 1;

    if (c != 0)
    {
        stream->bitBuffer |= 1;
    }

    /* write bit buffer if we have 8 bits */
    if (stream->bitCount == 8)
    {
        /* Printing in buffer */
        //	putInBuffer(stream->bitBuffer); 
        putInBufferStream(stream->bitBuffer,stream);

        /* reset buffer */
        stream->bitCount = 0;
        stream->bitBuffer = 0;
    }

    return returnValue;
}

/***************************************************************************
 *   Function   : BitFilePutBits
 *   Description: This function writes the specified number of bits from the
 *                memory location passed as a parameter to the file passed
 *                as a parameter.   Bits are written msb to lsb.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *   Effects    : Writes bits to the bit buffer and file stream.  The bit
 *                buffer will be modified as necessary.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
    __device__ inline static 
int BitFilePutBits(bit_file_t *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;

    if ((stream == NULL) || (bits == NULL))
    {
        return(EOF);
    }

    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitFilePutChar(bytes[offset], stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* write remaining bits */
        tmp = bytes[offset];

        while (remaining > 0)
        {
            returnValue = BitFilePutBit((tmp & 0x80), stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            tmp <<= 1;
            remaining--;
        }
    }

    return count;
}

/***************************************************************************
 *   Function   : BitFilePutBitsInt
 *   Description: This function provides a machine independent layer that
 *                allows a single function call to write an arbitrary number
 *                of bits from an integer type variable into a file.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *                size - sizeof type containing "bits"
 *   Effects    : Calls a function that writes bits to the bit buffer and
 *                file stream.  The bit buffer will be modified as necessary.
 *                the bits will be written to the file stream from least
 *                significant byte to most significant byte.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
    __device__ inline static 
int BitFilePutBitsInt(bit_file_t *stream, void *bits, const unsigned int count,
        const size_t size)
{
    int returnValue;

    if ((stream == NULL) || (bits == NULL))
    {
        return(EOF);
    }

    /* For now we assume our system is LITTLE ENDIAN */
    returnValue = BitFilePutBitsLE(stream, bits, count);

    return returnValue;
}

/***************************************************************************
 *   Function   : BitFilePutBitsLE   (Little Endian)
 *   Description: This function writes the specified number of bits from the
 *                memory location passed as a parameter to the file passed
 *                as a parameter.   Bits are written LSB to MSB.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *   Effects    : Writes bits to the bit buffer and file stream.  The bit
 *                buffer will be modified as necessary.  bits is treated as
 *                a little endian integer of length >= (count/8) + 1.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
    __device__ inline static 
int BitFilePutBitsLE(bit_file_t *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;
    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitFilePutChar(bytes[offset], stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* write remaining bits */
        tmp = bytes[offset];
        tmp <<= (8 - remaining);

        while (remaining > 0)
        {
            returnValue = BitFilePutBit((tmp & 0x80), stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            tmp <<= 1;
            remaining--;
        }
    }

    return count;
}

/* Freeing Bit Stream */
__device__ inline static 
void FreeBitStream(bit_file_t *stream){
    free(stream);
}

/* Printing our output buffer */
static 
void printBufferStream(char *buffer,int length) {
    int i;
    FILE *fp = NULL;
    if((fp = fopen(outFile,"wb")) == NULL) {
        printf("Couldn't open file. Will not print buffer\n");
    }
    printf("Printing Buffer\n");
    for ( i = 0 ; i < length ; i++){
        fputc(buffer[i],fp);
    }

    fclose(fp);
}

/* Printing our output buffer */
__device__ inline static 
void putInBufferStream(unsigned char c,bit_file_t * stream) {
    (stream->outBuffer)[stream->outBytes] = c;
    stream->outBytes++;
}

/**********END OF BITFILE FUNCTIONS**********************/

//namespace cuda 
//{
__global__ 
    void 
EncodeLZSSByArray(char *input,int input_len,char *output,int *output_length)
{
    __shared__ char sInput[MAX_BYTES_PER_BLOCK];
    int count ;
    bit_file_t *bfpOut;
    encoded_string_t matchData;
    unsigned int i, c;
    unsigned int len;                       /* length of string */
    /* head of sliding window and lookahead */
    unsigned int windowHead, uncodedHead;
    char *perThreadInput; 
    int perThreadInputLen;
    // copy input array into shared memory
    int tidWithBlock = threadIdx.x + blockIdx.x * MAX_THREADS_PER_BLOCK;
    
    // TODO: can optimize
    for(i=0; i < MAX_BYTES_PER_THREAD; i++) {
        sInput[threadIdx.x] =
            (tidWithBlock < input_len) ? input[tidWithBlock + i] : 0; 
    }
    perThreadInput = sInput + ((threadIdx.x) * MAX_BYTES_PER_THREAD);
    perThreadInputLen = MAX_THREADS_PER_BLOCK;
    perThreadOutput = output + (tidWithBlock * MAX_BYTES_PER_THREAD);

    windowHead = 0;
    uncodedHead = 0;

    /* Window Size : 2^12 same as offset  */
    /************************************************************************
     * Fill the sliding window buffer with some known vales.  DecodeLZSS must
     * use the same values.  If common characters are used, there's an
     * increased chance of matching to the earlier strings.
     ************************************************************************/
    memset(slidingWindow, ' ', WINDOW_SIZE * sizeof(unsigned char));

    bfpOut = BitStreamOpen((unsigned char *)perThreadOutput);

    count = 0;
    /* MAX_CODED : 2 to 17 because we cant have 0 to 1 */
    /************************************************************************
     * Copy MAX_CODED bytes from the input file into the uncoded lookahead
     * buffer.
     ************************************************************************/
    for (len = 0; len < MAX_CODED && (count < perThreadInputLen); len++)
    {
        c = perThreadInput[count];
        uncodedLookahead[len] = c;
        count++;
    }

    if (len == 0)
    {
        //return (EXIT_SUCCESS);   /* inFile was empty */
        return;
    }

    /* Look for matching string in sliding window */
    /*    InitializeSearchStructures(); Not needed for bruteforce */
    matchData = FindMatch(windowHead, uncodedHead);

    /* now encoded the rest of the file until an EOF is read */
    while (len > 0)
    {
        if (matchData.length > len)
        {
            /* garbage beyond last data happened to extend match length */
            matchData.length = len;
        }

        if (matchData.length <= MAX_UNCODED)
        {
            /* not long enough match.  write uncoded flag and character */
            BitFilePutBit(UNCODED, bfpOut);
            BitFilePutChar(uncodedLookahead[uncodedHead], bfpOut);

            matchData.length = 1;   /* set to 1 for 1 byte uncoded */
        }
        else
        {
            unsigned int adjustedLen;

            /* adjust the length of the match so minimun encoded len is 0*/
            adjustedLen = matchData.length - (MAX_UNCODED + 1);

            /* match length > MAX_UNCODED.  Encode as offset and length. */
            BitFilePutBit(ENCODED, bfpOut);
            BitFilePutBitsInt(bfpOut, &matchData.offset, OFFSET_BITS,
                    sizeof(unsigned int));
            BitFilePutBitsInt(bfpOut, &adjustedLen, LENGTH_BITS,
                    sizeof(unsigned int));
        }

        /********************************************************************
         * Replace the matchData.length worth of bytes we've matched in the
         * sliding window with new bytes from the input file.
         ********************************************************************/
        i = 0;
        while ((i < matchData.length) && (count < (perThreadInputLen + 1)))
        {
            c = perThreadInput[count];
            /* add old byte into sliding window and new into lookahead */
            ReplaceChar(windowHead, uncodedLookahead[uncodedHead]);
            uncodedLookahead[uncodedHead] = c;
            windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
            uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
            i++;
            count++;
        }

        /* handle case where we hit EOF before filling lookahead */
        while (i < matchData.length)
        {
            ReplaceChar(windowHead, uncodedLookahead[uncodedHead]);
            /* nothing to add to lookahead here */
            windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
            uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
            len--;
            i++;
        }

        /* find match for the remaining characters */
        matchData = FindMatch(windowHead, uncodedHead);
    }

    printf("Before writing to file\n");

    // TODO: Change this to work with multithread
    *output_length = bfpOut->outBytes;
    FreeBitStream(bfpOut);


}

void encode(char *input, int length, char *output)
{

    char *input_d;
    int size = (length + 1) * sizeof(char); //strlen + 1 space for NULL
    char *output_d;
    int *output_length_d;
    int output_length;

    int numBlocksToLaunch;
    int numThreadsToLaunch;
    int numThreadsInLastBlock;
    int bytesInLastBlock;
    /***************************************************
      1st Part: Allocation of memory on device memory  
     ****************************************************/
    unsigned char *slide;
    unsigned char *uncoded;

    cudaMalloc(&slide, WINDOW_SIZE * sizeof(unsigned char));
    cudaMemcpyToSymbol(slidingWindow, &slide , sizeof(slide));

    cudaMalloc(&uncoded, MAX_CODED * sizeof(unsigned char));
    cudaMemcpyToSymbol(uncodedLookahead, &uncoded , sizeof(uncoded));

    /* copy input matrix */
    cudaMalloc((void**) &input_d, size);
    cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);
    /*allocate memory for compressed output on device */
    cudaMalloc((void**) &output_d, size);
    /*allocate memory for compressed output on device */
    cudaMalloc((void**) &output_length_d, sizeof(int));
    
   
    /***************************************************
      2nd Part: Inovke kernel 
     ****************************************************/
    bytesInLastBlock = length % MAX_BYTES_PER_BLOCK;
    numBlocksToLaunch = (length / MAX_BYTES_PER_BLOCK) + (!!(bytesInLastBlock));
    numThreadsInLastBlock = (bytesInLastBlock / MAX_BYTES_PER_THREAD) + (!!(bytesInLastBlock % MAX_BYTES_PER_THREAD));
    numThreadsToLaunch = MAX_BYTES_PER_THREAD;
    if (numBlocksToLaunch <= 0) {
        printf("Error in calculating numBlocksToLaunch.");
        exit(-1);
    }

    EncodeLZSSByArray<<<numBlocksToLaunch, numThreadsToLaunch>>>(input_d,length,output_d,output_length_d);

    /***************************************************
      3rd Part: Transfer result from device to host 
     ****************************************************/
    cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&output_length, output_length_d, sizeof(int), cudaMemcpyDeviceToHost);
    printBufferStream(output,output_length);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(output_length_d);
}  
//} // namespace cuda
