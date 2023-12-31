10  'TRANSMIT : Analyzes Transmitted Ions from TRIM.
20  '---------------------------------------------------------
30  MAXNUMB=2000
40  DIM ENERGY(MAXNUMB),X,Y,Z
50  DIM COSX(MAXNUMB),ANGLE(MAXNUMB), LATERAL(MAXNUMB)
60  DIM E(100), A(100), L(100), N(100)
70  WIDTH 80: KEY OFF: COLOR 15: CLS: PRINT: LOCATE ,,0,0,8
80  PRINT "     浜様様様様様様様様様様様様様様様様様様様様様様融"
90  PRINT "     �   Analysis of Transmitted Ions from TRIM     �"
100 PRINT "     把陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳超":COLOR 11
110 PRINT "     �   This program analyzes the data in file     �"
120 PRINT "     �   TRANSMIT which contains Energy/Angle data. �"
130 PRINT "     把陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳陳超":COLOR 14
140 PRINT "     �   Show INPUT data while reading ?   (N)      �"
150 PRINT "     藩様様様様様様様様様様様様様様様様様様様様様様夕";
160 Y=CSRLIN : LOCATE Y-1,45,1
170 A$=INKEY$:IF A$="" GOTO 170 ELSE IF A$="Y" OR A$="y" THEN SKIP=1
180 LOCATE ,,0
190 OPEN "TRANSMIT.TXT" FOR INPUT AS #1
200 FOR I=1 TO 9: LINE INPUT#1, A$ : NEXT I     ' 8 Lines of Comments
210 LINE INPUT #1, HEADER$                      ' Header text
220 FOR I=1 TO 2: LINE INPUT#1, A$: NEXT I      ' 2 Lines of Comments
230 COLOR 12: IF SKIP > 0 GOTO 280 ELSE LOCATE 10,1,0
240     PRINT "               浜様様様様様様様様様様様融"
250     PRINT "               � Reading Line :         �"
260     PRINT "               藩様様様様様様様様様様様夕"
270 GOTO 330
280 LOCATE 10,1,0: COLOR 12
290   PRINT  " Ion  Atomic Energy     Depth     Lateral-Position          Atom Direction"
300   PRINT  " Numb Number  (eV)       X(A)      Y(A)       Z(A)       Cos(X)  Cos(Y) Cos(Z)"
310   PRINT  "----- ------ --------- ---------- ---------- ---------- ------- ------- -------"
320 TRIMOUT$="#####   ##   #.###^^^^ +.####^^^^ +.####^^^^ +.####^^^^ +.##### +.##### +.#####"
330 COLOR 14
340   FOR I=1 TO MAXNUMB
350     IF EOF(1)=-1 GOTO 430
360     A$=INPUT$(1,#1)                                      ' Initial "T" tag
370     INPUT#1, NUMB,Z1,ENERGY(I),X,Y,Z,COSX(I),COSY,COSZ  ' Save only COSX(I)
380     LATERAL(I)=(Y^2+Z^2)^.5
390     IF SKIP =0 THEN LOCATE 11,32: PRINT NUMB: GOTO 420
400     LOCATE 14,1,0
410     PRINT USING TRIMOUT$;NUMB,Z1,ENERGY(I),X,Y,Z,COSX(I),COSY,COSZ
420     NEXT I
430  CLOSE
440 NTOTAL=I-1
450  ANGLEMAX=0
460  FOR I=1 TO NTOTAL                              ' Find Averages
470    IF ABS(COSX(I)<.000001) THEN COSX(I)=.000001 ' Prevent division by zero
480    ANGLE(I) = 57.2958*ATN((ABS((1-COSX(I)^2)^0.5))/COSX(I))
490    IF ANGLE(I) > ANGLEMAX THEN ANGLEMAX = ANGLE(I)
500  NEXT I
510  COLOR 13 : LOCATE 15,,0
520  PRINT       "     浜様様様様様様様様様様様様様様様様様様様様様様融"
530  PRINT       "     �   This data shows a Maximum deflection of :  �": COLOR 14
540  PRINT USING "     �            ##.###  degrees .                 �";ANGLEMAX: COLOR 13
550  PRINT       "     �                                              �"
560  PRINT       "     �   Input DEGREE increments for Analysis :     �"
570  PRINT       "     �               [        ]                     �"
580  PRINT       "     藩様様様様様様様様様様様様様様様様様様様様様様夕";
590     IF ANGLEMAX < 1 THEN ANG=.1 : GOTO 630
600     IF ANGLEMAX < 2 THEN ANG=.25: GOTO 630
610     IF ANGLEMAX < 5 THEN ANG=.5 : GOTO 630
620     IF ANGLEMAX <10 THEN ANG=2! ELSE ANG=5
630  Y=CSRLIN-1 : LOCATE Y,24,0,0,8: COLOR 14: PRINT USING "##.##";ANG;
640  LOCATE Y,36,1: COLOR 14:  INPUT "",DANG$
650  IF VAL(DANG$)<=0 THEN DANG=ANG ELSE DANG=VAL(DANG$)
660 NBIN=INT(ANGLEMAX/DANG)
670  FOR J=1 TO NTOTAL
680     AA=CINT(ANGLE(J)/DANG): IF AA>100 THEN AA=100   'Angle Bin
690     A(AA)=A(AA)+1                                ' Ion Bin
700     N(AA)=N(AA)+1                                ' Ion Bin
710     E(AA)=E(AA)+ENERGY(J)/1000                   ' Energy Bin
720     L(AA)=L(AA)+LATERAL(J)                       ' Lateral Distance
730     ANGMEAN#=ANGMEAN#+ABS(ANGLE(J))              ' Mean Final Angle
740     ANG2SUM#=ANG2SUM#+ANGLE(J)*ANGLE(J)          ' Final Angle Straggle
750     EMEAN#=EMEAN#+ENERGY(J)/1000                 ' Mean Final Energy
760     E2SUM#=E2SUM#+ENERGY(J)*ENERGY(J)/1000000!        ' Final Energy Straggle
770   NEXT J
780 ANGMEAN#=ANGMEAN#/NTOTAL                         ' Mean Final Angle
790 ANGSTRAG=SQR(ABS((ANG2SUM#/NTOTAL)-ANGMEAN#^2))  ' Final Angle Straggle
800 EMEAN#=EMEAN#/NTOTAL                             ' Mean Final Energy
810 ESTRAG=SQR(ABS((E2SUM#/NTOTAL)-EMEAN#^2))        ' Final Energy Straggle
820  FOR J=0 TO NBIN
830     A(J)=A(J)/NTOTAL   'Mean Ions    vs. Angle
840     IF N(J)=0 GOTO 870
850     E(J)=E(J)/N(J)     'Mean Energy  vs. Angle
860     L(J)=L(J)/N(J)     'Mean Lateral vs. Angle
870     NEXT J
880 CLS
890  OPEN "TRANS" FOR OUTPUT AS #2
900  COLOR 14: PRINT "    Output is in file  ";: COLOR 12: PRINT "TRANS ."
910 COLOR 15: ION$=LEFT$(ION$,2)
920  PRINT         "浜様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様�"
930  PRINT         "�"+LEFT$(HEADER$+SPACE$(75),75)+"�"
940  PRINT         "麺様様様様様様様様様様様様様様様様様曜様様様様様様様様様様様様様様様様様様様�"
950  COLOR 11
960  PRINT   USING "� Mean Final Angle     =##.## degree � Mean Final Energy     =########, keV �";ANGMEAN#;EMEAN#
970  PRINT   USING "� Final Angle Straggle =##.## degree � Final Energy Straggle = #######, keV �";ANGSTRAG;ESTRAG
980  COLOR 15
990  PRINT         "藩様様様様様様様様様様様様様様様様様擁様様様様様様様様様様様様様様様様様様様�"
1000  PRINT#2,      "浜様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様�"
1010 PRINT#2,USING "�      Ion = \\,  Mass = ###.###,   Incident Energy = ####### keV           �";ION$;M1;E0KEV
1020 PRINT#2,      "麺様様様様様様様様様様様様様様様様様曜様様様様様様様様様様様様様様様様様様様�"
1030 PRINT#2,USING "� Mean Final Angle     =##.## degree � Mean Final Energy     =########, keV �";ANGMEAN#;EMEAN#
1040 PRINT#2,USING "� Final Angle Straggle =##.## degree � Final Energy Straggle = #######, keV �";ANGSTRAG;ESTRAG
1050 PRINT#2,      "藩様様様様様様様様様様様様様様様様様擁様様様様様様様様様様様様様様様様様様様�"
1060 COLOR 12
1070 PRINT   ">>>>>>> Averaged Values for Various Degrees of Deflection ----------"
1080 PRINT#2,">>>>>>> Averaged Values for Various Degrees of Deflection ----------"
1090  PRINT   "Deflection   Number    Mean      Mean YZ  Fraction    Fraction per"
1100  PRINT#2,"Deflection   Number    Mean      Mean YZ  Fraction    Fraction per"
1110  PRINT   "Angles         of      Energy    Lateral  of Ions     Unit Solid Angle"
1120  PRINT#2,"Angles         of      Energy    Lateral  of Ions     Unit Solid Angle"
1130  PRINT   "(deg)         Ions     (keV)      (A)                 (Rel. to Zero)    "
1140  PRINT#2,"(deg)         Ions     (keV)      (A)                 (Rel. to Zero)    "
1150  PRINT   "-----------  -------   --------  -------  ----------  --------------"
1160  PRINT#2,"-----------  -------   --------  -------  ----------  --------------"
1170  AREA0= 3.1416*(DANG/2)*(DANG/2)  'Area of diameter DANG degrees
1180 '' AREA0= 3.1416*.5*.5   'Area of 0.5 degrees
1190  FLUX0= A(0) / AREA0
1200 COLOR 15
1210  FOR K=0 TO NBIN
1220     PRINT USING   "##.##_-##.##"; K*DANG,(K+1)*DANG;
1230     PRINT#2, USING"##.##_-##.##"; K*DANG,(K+1)*DANG;
1240     PRINT USING" #######"; N(K);
1250     PRINT#2, USING" #######"; N(K);
1260     PRINT USING"########.##  "; E(K);
1270     PRINT#2, USING"########.##  "; E(K);
1280     PRINT USING"######  "; L(K);
1290     PRINT#2, USING"######  "; L(K);
1300     PRINT USING"####.#####  "; A(K);: FLUXTOT=FLUXTOT+A(K)
1310     PRINT#2, USING"####.#####  "; A(K);: FLUXTOT=FLUXTOT+A(K)
1320     AREA=3.1416*((K*DANG+DANG/2)^2-(K*DANG-DANG/2)^2)
1330 ''''AREA=3.1416*((K+.5)^2-(K-.5)^2): IF K=0 THEN AREA=AREA0
1340          IF K=0 THEN AREA=AREA0
1350    '------------------ Calculate RELATIVE Flux compared to central flux.
1360     IF FLUX0 <=0 THEN PRINT " Insufficient Data": GOTO 1400
1370     FLUX=A(K)/AREA/FLUX0
1380     PRINT USING"####.#####"; FLUX
1390     PRINT#2, USING"####.#####"; FLUX
1400   NEXT K
1410 CLOSE
1420  COLOR 14: PRINT "Output is in file  ";: COLOR 12: PRINT "TRANS ."
