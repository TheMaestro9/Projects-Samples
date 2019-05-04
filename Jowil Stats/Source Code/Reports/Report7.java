package Jowil.Reports;

import Jowil.Reports.Utils.CsvUtils;
import Jowil.Reports.Utils.TxtUtils;
import Jowil.Reports.Utils.WordUtils;
import Jowil.Reports.Utils.XlsUtils;
import Jowil.Statistics;
import Jowil.Utils;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFCellStyle;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.util.CellUtil;
import org.apache.poi.xwpf.usermodel.ParagraphAlignment;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFRun;
import org.jsoup.nodes.Document;
import com.lowagie.text.DocumentException;
import org.jsoup.Jsoup;
import sun.nio.cs.UTF_32LE;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Report7 extends Report {

    ArrayList<ArrayList<ArrayList<ArrayList<String>>>> formsTableStats ;
    String [] tableTitles = { "Hardest Questions" , "Easiest Questions" , "Bad Questions" } ;
    String goodJobMsg = "Good job! The test has no bad questions.";

    public Report7(){
       constructor();
    }


    public Report7 (String resoursesPath){
        super(resoursesPath) ;
        constructor();
    }
    private void constructor() {
        reportTitle = "Questions Insights Report" ;
        workSpacePath = reportsPath + "report7\\" ;
        templatePath = workSpacePath + "report7Template.html";
        pdfHtmlPath = workSpacePath+outputFileName+".html" ;
    }

    private String prepareGoodJobMsg(int formIndex){
        int numberOfForms = formsTableStats.size() ;
        String tempGoodJobMsg = goodJobMsg ;
        if(numberOfForms>1)
            tempGoodJobMsg = tempGoodJobMsg.replace("The test" , "Form "+ (formIndex+1)) ;
        return tempGoodJobMsg ;
    }

    private Document generatePdfHtml( ) throws IOException {
        File file = new File(templatePath);
        Document doc = Jsoup.parse(file, "UTF-8");

        updateTemplateFooter(doc); // updates the date of the footer to the current date


        final String pageBreakHtml= "<div class='page-break'></div>\n" ;
        String templateBodyHtml = doc.select("div#template").html() ;
        for ( int formIndex = 0 ; formIndex < formsTableStats.size() ; formIndex++ ) {
            ArrayList<ArrayList<ArrayList<String>>> formTables = formsTableStats.get(formIndex);
            if(formIndex > 0 )
                doc.select("div.page-break").last().after(templateBodyHtml);
            if(formsTableStats.size()>1) // more than one Form
                doc.select("div.divTitle").last().text(reportTitle + ": Form "+(formIndex+1) ) ;

            boolean noBadQuestions = false ;
            for (int tableIndex = 0; tableIndex < formTables.size(); tableIndex++) {
                ArrayList<ArrayList<String>> table = formTables.get(tableIndex);
                if(table.size()>0) {
                    String tableRowsHtml = createRowsHtml(table, ";grayRow", "");
                    doc.select("tr.table-header" + (tableIndex + 1)).last().after(tableRowsHtml);
                }
                else {
                    doc.select("table.table3").last().remove();
                    String tempGoodJobMsg = prepareGoodJobMsg(formIndex) ;
                    String goodJobMsgHtml = "<h3 class=\"good-job-message\">\n" +
                            tempGoodJobMsg +
                            "        </h3>" ;

                    doc.select("h2.table-title").last().after(goodJobMsgHtml) ;
                    noBadQuestions = true ;
                }
            }
            if(formIndex < formsTableStats.size()-1) {
                if(noBadQuestions)
                    doc.select("h3").last().after(pageBreakHtml);
                else
                    doc.select("table").last().after(pageBreakHtml);
            }
        }

        return doc ;
    }
    @Override
    public void generateHtmlReport() throws IOException {

        Document doc = generatePdfHtml();
        doc.select("div#footer").remove() ;
        writeHtmlFile(outputFormatsFolderPaths[ReportsHandler.HTML]+outputFileName+".html" , doc);

    }

    @Override
    public void generatePdfReport() throws IOException, DocumentException {

        Document doc  = generatePdfHtml();
        doc.select("table.table3").attr("style" , "margin-bottom:0px") ;
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath, outputFormatsFolderPaths[ReportsHandler.PDF]+outputFileName+".pdf");

    }

    private ArrayList<ArrayList<String>> getTableWithHeaders (ArrayList<ArrayList<String>> table , int tableIndex) {
        ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(table) ;
        ArrayList<String> headers = new ArrayList<>();
        if(tableIndex ==0 ) {
            headers.add("Question Name"); headers.add("Difficulty (0-10)");headers.add("Distractors") ;
            headers.add("Correct Response Percentage");

        }
        else if (tableIndex == 1) {
            headers.add("Question Name"); headers.add("Difficulty (0-10)");headers.add("Non Distractors") ;
            headers.add("Correct Response Percentage");
        }
        else if (tableIndex == 2 ) {
            headers.add("Question Name");  headers.add("Point Biserial"); headers.add("Smart Distractors");
            headers.add("total"); headers.add("Upper 25%"); headers.add("Lower 25%");
        }
        tableWithHeaders.add(0,headers);
        return  tableWithHeaders ;
    }

    private String generateCharSeparatedValuesString(char separator) {
        int CHP = 2;
        String outputCsv = "";

        for (int formIndex = 0; formIndex < formsTableStats.size(); formIndex++) {
            String form = "";
            if (formsTableStats.size() > 1)
                form = ": Form " + (formIndex + 1);


            ArrayList<ArrayList<ArrayList<String>>> statsTables = formsTableStats.get(formIndex);

            ArrayList<ArrayList<ArrayList<String>>> tempTables = new ArrayList<>(); // to calcPage With
            ArrayList<String> txtTables = new ArrayList<>();
            for (int tableIndex = 0; tableIndex < statsTables.size(); tableIndex++) {
                ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(statsTables.get(tableIndex), tableIndex);
                if(statsTables.get(tableIndex).size()==0) {
                    String GJMTxt = "";
                    GJMTxt+= tableTitles[tableIndex] + Utils.generatePattern(TxtUtils.newLine,2) ;
                    String tempGoodJobMsg = prepareGoodJobMsg(formIndex) ;
                    GJMTxt+= TxtUtils.generateTitleLine(tempGoodJobMsg,TxtUtils.calcTableWidth(tableWithHeaders , CHP) , 2) ;
                    txtTables.add(GJMTxt) ;
                }else {
                    tempTables.add(tableWithHeaders);
                    txtTables.add(CsvUtils.generateTable(tableWithHeaders, separator, tableTitles[tableIndex]));
                }
            }

            ////////////////////////// Start filling the report ///////////////////////////

            int pageWidth = CsvUtils.calcPageWidth(tempTables);

            if (formIndex > 0)
                outputCsv += Utils.generatePattern("*", pageWidth) + CsvUtils.NEW_LINE;

            String txtTitle = CsvUtils.generateTitleLine( reportTitle + form ,separator,
                    pageWidth, 2);

            outputCsv += txtTitle;

            outputCsv += CsvUtils.stackTablesV(txtTables, 2);
        }

        return outputCsv ;
    }

    @Override
    public void generateTxtReport() {

        int CHP = 2 ;
        String outputTxt = ""  ;

        for (int formIndex= 0 ; formIndex < formsTableStats.size() ; formIndex++) {
            String form = "";
            if(formsTableStats.size()>1)
                form = ": Form "+ (formIndex+1) ;


            ArrayList<ArrayList<ArrayList<String>>> statsTables = formsTableStats.get(formIndex);

            ArrayList<ArrayList<ArrayList<String>>> tempTables = new ArrayList<>(); // to calcPage With
            ArrayList<String> txtTables = new ArrayList<>();
            for (int tableIndex = 0; tableIndex < statsTables.size(); tableIndex++) {
                ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(statsTables.get(tableIndex), tableIndex);
                if(statsTables.get(tableIndex).size()>0) {
                    tempTables.add(tableWithHeaders);
                    txtTables.add(TxtUtils.generateTxtTableAlignCenter(tableWithHeaders, tableTitles[tableIndex], CHP, false));
                }else {
                    String GJMTxt = "";
                    GJMTxt+= tableTitles[tableIndex] + Utils.generatePattern(TxtUtils.newLine,2) ;
                    String tempGoodJobMsg = prepareGoodJobMsg(formIndex) ;
                    GJMTxt+= TxtUtils.generateTitleLine(tempGoodJobMsg,TxtUtils.calcTableWidth(tableWithHeaders , CHP) , 2) ;
                    txtTables.add(GJMTxt) ;
                }
            }

            ////////////////////////// Start filling the report ///////////////////////////

            int pageWidth  = TxtUtils.calcPageWidth(tempTables, CHP) ;

            if(formIndex>0)
                outputTxt += Utils.generatePattern("*" , pageWidth)+TxtUtils.newLine ;

            String txtTitle = TxtUtils.generateTitleLine(reportTitle + form ,
                    pageWidth, 2 ) ;

            outputTxt+= txtTitle ;

            outputTxt+= TxtUtils.stackTablesV(txtTables, 2);
        }

        TxtUtils.writeTxtToFile(outputTxt , outputFormatsFolderPaths[ReportsHandler.TXT]+outputFileName+".txt");
    }

    @Override
    public void generatePrintablePdfReport() throws IOException, DocumentException {

        Document doc = generatePdfHtml() ;
        doc.select("table.table3").attr("style" , "margin-bottom:0px") ;
        styleTitlePrintable(doc);
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath , outputFormatsFolderPaths[ReportsHandler.PRINTABLE_PDF]+outputFileName+".pdf");
    }

    @Override
    public void generateCsvReport() throws IOException {

        String outputCsv = generateCharSeparatedValuesString(',') ;
        CsvUtils.writeCsvToFile(outputCsv , outputFormatsFolderPaths[ReportsHandler.CSV]+outputFileName+".csv");

    }

    @Override
    public void generateTsvReprot() {
        String outputCsv = generateCharSeparatedValuesString('\t') ;
        CsvUtils.writeCsvToFile(outputCsv , outputFormatsFolderPaths[ReportsHandler.TSV]+outputFileName+".tsv");
    }

    @Override
    public void generateWordReport() throws IOException, InvalidFormatException {

        XWPFDocument document = WordUtils.createDocument() ; // landscape size

        WordUtils.createWordFooter(document); ;

        int tableTitleFontSize = 18 ;

        for(int formIndex = 0 ; formIndex < formsTableStats.size() ; formIndex++) {
            String title = reportTitle;
            if( formsTableStats.size() >1) {
                title =   title +": Form " + (formIndex+1);
            }
            if(formIndex>0)
                WordUtils.addPageBreak(document);
            WordUtils.addTitle(document , title);


            ArrayList<ArrayList<ArrayList<String>>> statsTables = formsTableStats.get(formIndex);

            for(int tableIndex = 0 ; tableIndex<statsTables.size() ; tableIndex++) {
                if(tableIndex==2 && formIndex==0 )
                    WordUtils.addPageBreak(document);
                    
                ArrayList<ArrayList<String>> table = statsTables.get(tableIndex);
                if(table.size()>0) {
                    ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(table, tableIndex);
                    WordUtils.addTable(document, tableWithHeaders, WordUtils.TABLE_ALIGN_CENTER, tableTitles[tableIndex], tableTitleFontSize, true,true);
                }
                else {
                    XWPFParagraph tableTitlePar = document.createParagraph();
                    XWPFRun tableTitleRun = tableTitlePar.createRun() ;
                    tableTitleRun.setText(tableTitles[tableIndex]);
                    tableTitleRun.setFontSize(tableTitleFontSize);

                    String tempGoodJobMsg = prepareGoodJobMsg(formIndex) ;
                    XWPFParagraph par = document.createParagraph() ;
                    par.setAlignment(ParagraphAlignment.CENTER);
                    XWPFRun run = par.createRun();
                    run.setText(tempGoodJobMsg);
                    run.setBold(true);
                    run.setColor("008000");
                    run.setFontSize(15);
                }
            }
        }
        WordUtils.writeWordDocument(document , outputFormatsFolderPaths[ReportsHandler.WORD]+outputFileName+".docx");

    }
    @Override
    public void generateXlsReport() throws IOException {


        int pageWidth = 8;
        XlsUtils.createXls(pageWidth);

        HSSFSheet sheet = XlsUtils.sheet ;


        for (int formIndex = 0; formIndex < formsTableStats.size(); formIndex++) {
            String form = "";
            if (formsTableStats.size() > 1)
                form = ": Form " + (formIndex + 1);

            XlsUtils.addTitle(reportTitle+form);

            ArrayList<ArrayList<ArrayList<String>>> statsTables = formsTableStats.get(formIndex);

            for (int tableIndex = 0; tableIndex < statsTables.size(); tableIndex++) {
                ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(statsTables.get(tableIndex), tableIndex);
                if(statsTables.get(tableIndex).size()==0) {
                    XlsUtils.addTableTitle(tableTitles[tableIndex]);
//                    HSSFRow row = sheet.createRow(XlsUtils.lastRowIndex++) ;
//                    row.createCell(XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX).setCellValue(tableTitles[tableIndex]) ;
                    String GoodJobMsg = prepareGoodJobMsg(formIndex) ;
                    String[] parts =GoodJobMsg.split("no");

                    HSSFRow row  = sheet.createRow(XlsUtils.lastRowIndex++) ;
                    for(int i = 0 ; i < parts.length ; i++) {
                        String no = i == 0? "no":"" ;
                        HSSFCell cell  = row.createCell(XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX + i);
                        cell.setCellValue(parts[i]+no);
                        CellUtil.setFont(cell , XlsUtils.boldFont);
                    }
                    XlsUtils.lastRowIndex += XlsUtils.DEFAULT_NUMBER_OF_LINES_AFTER_TABLE;
                }else {
                    XlsUtils.addTableAlignCenter(tableWithHeaders , XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX,
                            tableTitles[tableIndex],XlsUtils.DEFAULT_NUMBER_OF_LINES_AFTER_TABLE);
                }
            }


        }

        XlsUtils.writeXlsFile(outputFormatsFolderPaths[ReportsHandler.XLS]+outputFileName+".xls" );

    }

    @Override
    public void init() {
        formsTableStats = Statistics.report7Stats() ;
    }
}
