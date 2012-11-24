/***************************************************************************
*                     Sample Program Using LZSS Library
*
*   File    : sample.c
*   Purpose : Demonstrate usage of LZSS library
*   Author  : Michael Dipperstein
*   Date    : February 21, 2004
*
****************************************************************************
*   UPDATES
*
*   Revision 1.1  2004/02/22 17:36:30  michael
*   Initial revision.  Mostly code form old lzss.c.
*
*   11/07/04    Name changed to sample.c
*
*   $Id: sample.c,v 1.5 2007/09/20 04:34:45 michael Exp $
*   $Log: sample.c,v $
*   Revision 1.5  2007/09/20 04:34:45  michael
*   Replace getopt with optlist.
*   Changes required for LGPL v3.
*
*   Revision 1.4  2006/12/26 04:09:09  michael
*   Updated e-mail address and minor text clean-up.
*
*   Revision 1.3  2004/11/13 22:51:01  michael
*   Provide distinct names for by file and by name functions and add some
*   comments to make their usage clearer.
*
*   Revision 1.2  2004/11/11 14:37:26  michael
*   Open input and output files as binary.
*
*   Revision 1.1  2004/11/08 05:54:18  michael
*   1. Split encode and decode routines for smarter linking
*   2. Renamed lzsample.c sample.c to match my other samples
*   3. Makefile now builds code as libraries for better LGPL compliance.
*
*
****************************************************************************
*
* SAMPLE: Sample usage of LZSS Library
* Copyright (C) 2004, 2006, 2007 by
* Michael Dipperstein (mdipper@alumni.engr.ucsb.edu)
*
* This file is part of the lzss library.
*
* The lzss library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The lzss library is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
* General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
***************************************************************************/

/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lzss.h"
#include "optlist.h"

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/
typedef enum
{
    ENCODE,
    DECODE
} MODES;


/****************************************************************************
*   Function   : main
*   Description: This is the main function for this program, it validates
*                the command line input and, if valid, it will either
*                encode a file using the LZSS algorithm or decode a
*                file encoded with the LZSS algorithm.
*   Parameters : argc - number of parameters
*                argv - parameter list
*   Effects    : Encodes/Decodes input file
*   Returned   : EXIT_SUCCESS for success, otherwise EXIT_FAILURE.
****************************************************************************/
int main(int argc, char *argv[])
{
    option_t *optList, *thisOpt;
    FILE *fpIn, *fpOut;      /* pointer to open input & output files */
    MODES mode;

    char input[] = "This is a dummy string.Hello hey hey hey heye hye";
    int input_len = strlen(input);
    char *output = (char *)malloc(sizeof(char) * (input_len + 1));

    /* initialize data */
    fpIn = NULL;
    fpOut = NULL;
    mode = ENCODE;

    /* parse command line */
    optList = GetOptList(argc, argv, "c:h?");
    thisOpt = optList;

    while (thisOpt != NULL)
    {
        switch(thisOpt->option)
        {
            case 'c':       /* compression mode */
                mode = ENCODE;
                break;

            case 'h':
            case '?':
                printf("options:\n");
                printf("  -c : Encode input file to output file.\n");
                printf("  -d : Decode input file to output file.\n");
                printf("  -i <filename> : Name of input file.\n");
                printf("  -o <filename> : Name of output file.\n");
                printf("  -h | ?  : Print out command line options.\n\n");
                FreeOptList(optList);
                return(EXIT_SUCCESS);
        }

        optList = thisOpt->next;
        free(thisOpt);
        thisOpt = optList;
    }



    /* we have valid parameters encode or decode */
    if (mode == ENCODE)
    {
        printf("Before encoding\n");
        EncodeLZSSByArray(input, output);
    }

    return EXIT_SUCCESS;
}
