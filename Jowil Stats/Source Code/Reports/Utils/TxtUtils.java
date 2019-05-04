package Jowil.Reports.Utils;

import Jowil.Translator;
import Jowil.Utils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import static org.apache.commons.math3.stat.StatUtils.sum;

public class TxtUtils {

    public static String newLine = "\r\n" ;

    /**
     * calculate max number of characters in table for each line
     * @param table table data as matrix
     * @param cellHorizontalPadding number of empty spaces in table cell
     * @return list of max number of characters needed for each line
     */
    public static ArrayList<Integer> calcCellsWidths ( ArrayList<ArrayList<String>>  table , int cellHorizontalPadding ) {
        ArrayList<Integer> cellsWidths = new ArrayList<>();
        for (int colIndex = 0; colIndex < table.get(0).size(); colIndex++) {
            int colMaxLength = 0;
            for (int rowIndex = 0; rowIndex < table.size(); rowIndex++) {
                int tableCellLenght = table.get(rowIndex).get(colIndex).length();
                if (tableCellLenght > colMaxLength)
                    colMaxLength = tableCellLenght;
            }
            cellsWidths.add(colMaxLength + cellHorizontalPadding * 2);
        }
        return cellsWidths;
    }

    public static int calcTableWidth(ArrayList<ArrayList<String>> table , int cellHorizontalPadding){
      return (int) sum(calcCellsWidths(table , cellHorizontalPadding).stream().mapToDouble(d -> d).toArray());
    }

    public static int calcPageWidth (ArrayList<ArrayList<ArrayList<String>>> pageTables , ArrayList<Integer> tablesCHP) {
        int pageWidth = 0  ;
        for(int tableIndex = 0 ; tableIndex < pageTables.size() ; tableIndex++) {
            ArrayList<ArrayList<String>> table = pageTables.get(tableIndex)  ;
            int tableWidth = calcTableWidth(table , tablesCHP.get(tableIndex));
            if(tableWidth>pageWidth)
                pageWidth = tableWidth ;
        }
        return pageWidth ;
    }

    /**
     * calculates max number of characters in a single line in all page tables
     * @param pageTables list of tables data
     * @param CHP cell horizontal padding
     * @return the max number of characters
     */
    public static int calcPageWidth (ArrayList<ArrayList<ArrayList<String>>> pageTables , int CHP) {
      ArrayList<Integer> CHPs = new ArrayList<>();
        for(int i = 0 ; i < pageTables.size() ; i++){
            CHPs.add(CHP) ;
        }
        return  calcPageWidth(pageTables , CHPs) ;
    }

    public static String generateTitleLine (String title , int pageWidth , int paddingBelow) {
        String padding = Utils.generatePattern(newLine , paddingBelow) ;
        String spaceBeofore =   Utils.generatePattern( " " , (int)Math.floor((pageWidth-title.length())/2)) ;
        return spaceBeofore+ title + padding ;
    }

    public static String generateTxtTableAlignCenter (ArrayList<ArrayList<String>> table , String title , int cellHorizontalPadding , boolean showLines ) {
        return generateTxtTableAlignCenter(table, title,cellHorizontalPadding,showLines,false) ;
    }

    /**
     * generates String representing a table with data aligned center in each cell
     * @param table table data as matrix
     * @param title title to be written above table
     * @param cellHorizontalPadding number of empty spaces in table cell
     * @param showLines boolean to show lines between rows of tables
     * @param arabic boolean if text contains arabic text
     * @return String representing table
     */
    public static String generateTxtTableAlignCenter (ArrayList<ArrayList<String>> table , String title , int cellHorizontalPadding , boolean showLines ,boolean arabic ) {
        if(table.size()>1&& Utils.checkListContainArabic(table.get(1)) ) {
            table = Translator.translateTable(table) ;
        }
        String txtTable = "" ;
        ArrayList<Integer> cellsWidths = calcCellsWidths(table , cellHorizontalPadding) ;
        int tableWidth = calcTableWidth( table ,  cellHorizontalPadding) ;
        if(title!="")
            txtTable+=title+newLine;
        txtTable+= Utils.generatePattern("—" , tableWidth)  +newLine  ;
        for ( int rowIndex = 0 ; rowIndex < table.size() ;rowIndex++ ) {
            for( int colIndex = 0 ; colIndex < table.get(1).size() ; colIndex++) {
                String cellString = table.get(rowIndex).get(colIndex) ;
                int numberOfSpacesNeeded = (int)Math.ceil((double)(cellsWidths.get(colIndex)- cellString.length())/2) ;
                int numberOfSpacesNeeded2 = 0 ;
                if(colIndex!=0) {
                    String prevString =  table.get(rowIndex).get(colIndex-1) ;
                    numberOfSpacesNeeded2 += Math.floor((cellsWidths.get(colIndex - 1)  - prevString.length()) / 2) ;
                }
                if(arabic)
                    txtTable +=Utils.generatePattern(" " , numberOfSpacesNeeded2) +"I" + Utils.generatePattern(" ",numberOfSpacesNeeded) + cellString ;
                else
                    txtTable +=Utils.generatePattern(" " , numberOfSpacesNeeded2 + numberOfSpacesNeeded) + cellString ;

            }
            // add line after the first Row
            if (rowIndex ==0 || showLines)
                txtTable+=newLine + Utils.generatePattern("—" , tableWidth)    ;

            txtTable+=newLine ;

        }
        return txtTable ;
    }

    public static String generateTxtTableAlignCenter2 (ArrayList<ArrayList<String>> table , String title , int cellHorizontalPadding , boolean showLines ) {

        String txtTable = "" ;
        ArrayList<Integer> cellsWidths = calcCellsWidths(table , cellHorizontalPadding) ;
        int tableWidth = calcTableWidth( table ,  cellHorizontalPadding) ;
        if(title!="")
            txtTable+=title+newLine;
        txtTable+= Utils.generatePattern("—" , tableWidth)  +newLine  ;
        for ( int rowIndex = 0 ; rowIndex < table.size() ;rowIndex++ ) {
            for( int colIndex = 0 ; colIndex < table.get(1).size() ; colIndex++) {
                String cellString = table.get(rowIndex).get(colIndex) ;
                int numberOfSpacesNeeded = (int)Math.ceil((double)(cellsWidths.get(colIndex)- cellString.length())/2) ;
                if(colIndex!=0) {
                    String prevString =  table.get(rowIndex).get(colIndex-1) ;
                    numberOfSpacesNeeded += Math.floor((cellsWidths.get(colIndex - 1)  - prevString.length()) / 2) ;
                }
                txtTable += Utils.generatePattern(" ",numberOfSpacesNeeded) + cellString ;
            }
            // add line after the first Row
            if (rowIndex ==1 || showLines)
                txtTable+=newLine + Utils.generatePattern("—" , tableWidth)    ;

            txtTable+=newLine ;

        }
        return txtTable ;
    }

    /**
     * generates String representing a table with data aligned left for even cell index and right for odd cell index
     * @param table table data as matrix
     * @param title title to be written above table
     * @param cellHorizontalPadding number of empty spaces in table cell
     * @return
     */
    public static String generateTxtTableAlignLR (ArrayList<ArrayList<String>> table , String title , int cellHorizontalPadding ) {
        String txtTable = "" ;
        int paddingBetweenTables = 5 ;
        ArrayList<Integer> cellsWidths = calcCellsWidths(table , cellHorizontalPadding) ;
        int tableWidth = calcTableWidth( table ,  cellHorizontalPadding) ;
        if(title!="")
            txtTable+= title +newLine ;
        txtTable+= Utils.generatePattern("_" , (int)tableWidth)  +newLine  ;
        for ( int rowIndex = 0 ; rowIndex < table.size() ;rowIndex++ ) {
            for( int colIndex = 0 ; colIndex < table.get(1).size() ; colIndex++) {
                String cellString = table.get(rowIndex).get(colIndex) ;
                int numberOfSpacesNeeded = 0 ;
                if(colIndex%2==0) {
                    if(colIndex > 0)
                        numberOfSpacesNeeded+=paddingBetweenTables ;
//                    txtTable+= "    |";
                }
                else {
                    String prevString =  table.get(rowIndex).get(colIndex-1) ;
                    numberOfSpacesNeeded+= cellsWidths.get(colIndex) + cellsWidths.get(colIndex-1) - prevString.length() -cellString.length()-paddingBetweenTables ;
                }
                txtTable += Utils.generatePattern(" ",numberOfSpacesNeeded) + cellString ;
            }
            // add line after the first Row
//            if (rowIndex ==0)
                txtTable+=newLine + Utils.generatePattern("-" , tableWidth)    ;

            txtTable+=newLine ;

        }
        return txtTable ;
    }

    /**
     * Stack Tables after bing converted to txt string Vertically
     * @param tables List of  strings representing tables
     * @param paddingBetweenTables number of lines that will separate between tables
     * @return String representing the stacked tables
     */
    public static String stackTablesV (ArrayList<String> tables , int paddingBetweenTables ) {
        String outputTable = "";
        for(String table : tables) {
            outputTable += table ;
            outputTable += Utils.generatePattern(newLine , paddingBetweenTables) ;
        }

        return outputTable ;
    }

    public static void writeTxtToFile(String txt , String fileName )
    {
        try {
            PrintWriter out = new PrintWriter(fileName);
            out.println(txt);
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


    }
}
