10      COLOR 15,0: CLS
20  PRINT "浜様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様融
30  PRINT "�       SUM  =  Sums up ASCII outputs of TRIM calculation.       �
40  PRINT "麺様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様郵
50  PRINT "�  (C)-1989, J. F. Ziegler, IBM-Research, Yorktown, NY 10598     �
60  PRINT "麺様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様郵
70  C1$=  "�  Reading file VACANCY.TXT -- Distribution of Vacancies         �
80  C2$=  "�  Reading file IONIZ.TXT   -- Distribution of Ionization        �
90  C3$=  "�  Reading file PHONON.TXT  -- Distribution of Phonons           �
100 C4$=  "�  Reading file E2RECOIL.TXT-- Distribution of Energy to Recoils.�
110 C5$=  "藩様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様様夕
120         DIM DEPTH(100), VAC1(100), VAC2(100), IONIZ1(100), IONIZ2(100)
130         DIM PHONON1(100), PHONON2(100), E2REC1(100), E2REC2(100)
140 '
150 '様様様様様様様様様� Getting VACANCY.TXT  様様様様様様様様様様
160     OPEN "VACANCY.TXT" FOR INPUT AS #1
170  PRINT C1$
180  J=1                                      ' This input style is used to
190  FOR I=1 TO 200                           ' get rid of Comments in file.
200     A=EOF(1): IF A=-1 GOTO 250
210     LINE INPUT#1,A$
220     A=VAL(MID$(A$,1,11)): B=VAL(MID$(A$,13,12)): C=VAL(MID$(A$,26,12))
230     IF A>0 THEN  DEPTH(J)=A: VAC1(J)=B: VAC2(J)=C: J=J+1
240  NEXT I :
250  CLOSE #1
260 '様様様様様様様様様� Getting IONIZ.TXT  様様様様様様様様様様
270  OPEN "IONIZ.TXT" FOR INPUT AS #1
280  PRINT C2$
290  J=1                                      ' This input style is used to
300  FOR I=1 TO 200                           ' get rid of Comments in file.
310     A=EOF(1): IF A=-1 GOTO 360
320     LINE INPUT#1,A$
330     A=VAL(MID$(A$,1,11)): B=VAL(MID$(A$,13,12)): C=VAL(MID$(A$,26,12))
340     IF A>0 THEN  DEPTH(J)=A: IONIZ1(J)=B: IONIZ2(J)=C: J=J+1
350  NEXT I :
360  CLOSE #1
370 '様様様様様様様様様� Getting PHONON.TXT  様様様様様様様様様様
380     OPEN "PHONON.TXT" FOR INPUT AS #1
390  PRINT C3$
400  J=1                                      ' This input style is used to
410  FOR I=1 TO 200                           ' get rid of Comments in file.
420     A=EOF(1): IF A=-1 GOTO 470
430     LINE INPUT#1,A$
440     A=VAL(MID$(A$,1,11)): B=VAL(MID$(A$,13,12)): C=VAL(MID$(A$,26,12))
450     IF A>0 THEN  DEPTH(J)=A: PHONON1(J)=B: PHONON2(J)=C: J=J+1
460  NEXT I :
470  CLOSE #1
480 '様様様様様様様様様� Getting E2RECOIL.TXT  様様様様様様様様様様
490     OPEN "E2RECOIL.TXT" FOR INPUT AS #1     'Energy Transferred to Recoils
500  PRINT C4$
510  J=1                                      ' This input style is used to
520  FOR I=1 TO 200                           ' get rid of Comments in file.
530     A=EOF(1): IF A=-1 GOTO 580
540     LINE INPUT#1,A$
550     A=VAL(MID$(A$,1,11)): B=VAL(MID$(A$,13,12)): C=VAL(MID$(A$,26,12))
560     IF A>0 THEN  DEPTH(J)=A: E2REC1(J)=B: E2REC2(J)=C: J=J+1
570  NEXT I :
580  CLOSE #1
590  PRINT C5$
600  COLOR 13  '烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝烝�
610     INPUT "What was the Original Ion Energy (keV) ?   [100]  "; E0KEV
620     IF E0KEV = 0 THEN E0KEV=100
630     INPUT "What is the Target Binding Energy (eV) ?   [  2]"; EB
640     IF EB = 0 THEN EB=2
650     CW=DEPTH(2)-DEPTH(1)
660     FOR I= 1 TO 100
670        VAC1   = VAC1+VAC1(I)      : VAC2   = VAC2+VAC2(I)
680        IONIZ1 = IONIZ1+IONIZ1(I)  : IONIZ2 = IONIZ2+IONIZ2(I)
690        PHONON1= PHONON1+PHONON1(I): PHONON2= PHONON2+PHONON2(I)
700        E2REC1 = E2REC1+E2REC1(I)  : E2REC2 = E2REC2+E2REC2(I)
710        NEXT I
720     VAC1   =VAC1*CW*EB: VAC2=VAC2*CW*EB
730     IONIZ1 =IONIZ1*CW : IONIZ2=IONIZ2*CW
740     PHONON1=PHONON1*CW: PHONON2=PHONON2*CW
750     E2REC1 =E2REC1*CW : E2REC2=E2REC2*CW
760     TOTLOSS = VAC1+VAC2+IONIZ1+IONIZ2+PHONON1+PHONON2
770     TOTAL = E0KEV*1000
780  COLOR 14
790     PRINT      "浜様様様様様様様様様様様様様曜様様様様様様様様様様様様融
800     PRINT      "�           PERCENTAGE LOSS  �     ENERGY LOSS (keV)   �"
810     PRINT      "�           IONS    RECOILS  �    IONS      RECOILS    �"
820     PRINT      "�          陳陳陳�  陳陳陳�  �  陳陳陳陳陳 陳陳陳陳陳  �"
830     I1=IONIZ1*100/TOTLOSS : I2=IONIZ2*100/TOTLOSS
840     V1=VAC1*100/TOTLOSS   : V2=VAC2*100/TOTLOSS
850     P1=PHONON1*100/TOTLOSS: P2=PHONON2*100/TOTLOSS
860     PRINT USING"� Ioniz =   ##.###   ##.###  �  ######.### ######.###  �";I1;I2,IONIZ1/1000;IONIZ2/1000
870     PRINT USING"� Vac.  =   ##.###   ##.###  �  ######.### ######.###  �";V1;V2;VAC1/1000;VAC2/1000
880     PRINT USING"� Phonon=   ##.###   ##.###  �  ######.### ######.###  �";P1;P2;PHONON1/1000;PHONON2/1000
890     PRINT      "� 様様様�  様様様�  様様様�  �  様様様様様 様様様様様  �"
900     TOTAL1=(IONIZ1+VAC1+PHONON1)*100/TOTLOSS
910     TOTAL2=(IONIZ2+VAC2+PHONON2)*100/TOTLOSS
920     TOTAL3=(IONIZ1+VAC1+PHONON1)/1000
930     TOTAL4=(IONIZ2+VAC2+PHONON2)/1000
940     PRINT USING"� Totals=  ###.###% ###.###% �  ######.### ######.###  �";TOTAL1;TOTAL2;TOTAL3;TOTAL4
950     PRINT      "藩様様様様様様様様様様様様様擁様様様様様様様様様様様様夕"
960     PRINT      "        浜様様様様様様様様様様様様様様様様様様様�"
970     PRINT USING"        �   TOTAL ENERGY LOSS = ######.### keV  �";TOTLOSS/1000
980     PRINT      "        藩様様様様様様様様様様様様様様様様様様様�"
990     VAC1=CINT(VAC1/EB) : VAC2=CINT(VAC2/EB)
1000     PRINT      "浜様様様様様様様様様様様様様様様様様様様様様様様様様様�"
1010     PRINT USING"�  Energy Transferred to Recoils =######.### keV      �";E2REC1/1000
1020 ''' PRINT USING"�  Energy Received by Recoils    =######.### keV      �";E2REC2/1000
1030     PRINT USING"�  Ion Vacancies = ###### , Recoil Vacancies = ###### �";VAC1;VAC2
1040     PRINT USING"�  Total Vacancies/Ion =  #######                     �";VAC1+VAC2
1050     PRINT      "藩様様様様様様様様様様様様様様様様様様様様様様様様様様�"
1060  COLOR 15: PRINT "NOTE : Totals may be in error if final ion is not yet stopped,"
1070  COLOR 15: PRINT "       or if Ions or Recoils leave the Plotting Wondow."
1080  COLOR 15: PRINT "NOTE : Summation currently valid only for single element targets."
