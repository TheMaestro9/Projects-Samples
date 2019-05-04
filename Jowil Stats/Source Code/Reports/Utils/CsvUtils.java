package Jowil.Reports.Utils;

import Jowil.Translator;
import Jowil.Utils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

public class CsvUtils {

    private static final char DEFAULT_SEPARATOR = ',';
    public static final String NEW_LINE = "\r\n" ;

    public static String generateLine(ArrayList<String> values)  {
        return generateLine(values, DEFAULT_SEPARATOR, ' ');
    }

    public static String generateLine(ArrayList<String> values, char separators) {
        return generateLine(values, separators, ' ');
    }

    //https://tools.ietf.org/html/rfc4180
    private static String followCVSformat(String value) {

        String result = value;
        if (result.contains("\"")) {
            result = result.replace("\"", "\"\"");
        }
        return result;

    }

    /**
     * Generate one line (row) of csv String
     * @param values list of values in the line
     * @param separators separator used in csv
     * @param customQuote optional char that wrapp values
     * @return String representing Csv line
     */
    public static String generateLine(ArrayList<String> values, char separators, char customQuote) {

        boolean first = true;

        //default customQuote is empty

        if (separators == ' ') {
            separators = DEFAULT_SEPARATOR;
        }

        StringBuilder sb = new StringBuilder();
        for (String value : values) {
            if (!first) {
                sb.append(separators);
            }
            if (customQuote == ' ') {
                sb.append(followCVSformat(value));
            } else {
                sb.append(customQuote).append(followCVSformat(value)).append(customQuote);
            }

            first = false;
        }
        sb.append(NEW_LINE);

        return sb.toString();
    }


    /**
     * Function that generate CSV String That Represent a table
     * @param table table data as Matrix
     * @param separator separator used in CSV
     * @param title Tilte that will be written above the table  DEFAULT ""
     * @return
     */

    public static  String generateTable(ArrayList<ArrayList<String>> table, char separator , String title)  {

        if(table.size()>1&& Utils.checkListContainArabic(table.get(1)) ) {
            table = Translator.translateTable(table) ;
        }
        String tableCsv = "";
        if(!title.equals(""))
            tableCsv += title+NEW_LINE ;
        for (ArrayList<String> tableRow : table)
            tableCsv += generateLine(tableRow, separator,'"');
        return tableCsv;
    }
    public static  String generateTable(ArrayList<ArrayList<String>> table, char separator)  {
        return generateTable(table , separator , "");
    }


    public static String generateTitleLine (String title , char separator , int pageWidth ,int paddingBelow){
        int numberOfTitleWords = title.split(" ").length;
       return Utils.generatePattern(String.valueOf(separator), pageWidth/2-numberOfTitleWords/2)+title+Utils.generatePattern(NEW_LINE , paddingBelow);
    }

    public static int calcTableWidth (ArrayList<ArrayList<String>> table){
        return table.get(0).size() ;
    }

    /**
     * calculates number of cells (horizontally) needed by tables
     * @param tables tables in the page
     * @return number of cells needed
     */
    public static int calcPageWidth(ArrayList<ArrayList<ArrayList<String>>> tables) {
        int maxWidth = 0 ;
        for (ArrayList<ArrayList<String>> table :tables){
            int tableWidth = calcTableWidth(table) ;
            if(tableWidth>maxWidth)
                maxWidth = tableWidth;
        }
        return maxWidth ;
    }

    /**
     * Stack Tables after bing converted to csv string Vertically
     * @param tables List of CSV strings representing tables
     * @param paddingBetweenTables number of lines that will separate between tables
     * @return String representing the stacked tables
     */
    public static String stackTablesV (ArrayList<String> tables , int paddingBetweenTables ) {
        String outputTable = "";
        for(String table : tables) {
            outputTable += table ;
            outputTable += Utils.generatePattern( NEW_LINE , paddingBetweenTables) ;
        }

        return outputTable ;
    }



    public static void writeCsvToFile(String csv, String fileName) {
        try {
            PrintWriter out = new PrintWriter(fileName);
            out.println(csv);
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


    }
}