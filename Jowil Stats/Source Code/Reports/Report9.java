package Jowil.Reports;

import Jowil.CSVHandler;
import Jowil.Group;
import Jowil.Reports.Utils.CsvUtils;
import Jowil.Reports.Utils.TxtUtils;
import Jowil.Reports.Utils.WordUtils;
import Jowil.Reports.Utils.XlsUtils;
import Jowil.Statistics;
import Jowil.Utils;
import com.lowagie.text.DocumentException;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.xwpf.usermodel.*;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Report9 extends Report {

    private ArrayList<ArrayList<ArrayList<ArrayList<String>>>> groupsStatsTables ;
//    String imagesFolderFullPath ;

    public Report9(){
        constructor();
    }

    public Report9 (String resoursesPath){
        super(resoursesPath) ;
        constructor();
    }
    private void constructor() {
        reportTitle = "Questionnaire Report" ;
        workSpacePath = reportsPath + "report9\\" ;
        templatePath = workSpacePath + "report9Template.html";
        pdfHtmlPath = workSpacePath + outputFileName + ".html";

    }

    private Document generatePdfHtml (ArrayList<ArrayList<ArrayList<ArrayList<String>>>> groupsTables) throws IOException {
        File file = new File(templatePath);
        Document doc =  Jsoup.parse(file , "UTF-8") ;


        updateTemplateFooter(doc); // updates the date of the footer to the current date

        String wrapperHtml = doc.select("div.wrapper").outerHtml() ;
        String pageBreakHtml = "<div class='page-break'> </div>" ;
        String templateHtml = doc.select("div#template").html() ;
        String groupNameHtml = doc.select("div.line").outerHtml() ;

        final int  NUMBER_OF_ROWS_IN_PAGE = 50 ;
        final int  ROWS_OF_MAIN_HEADER = 12 ;
        final double  ROWS_OF_TABLE_HEADER = 5 ;
        final int  ROWS_OF_GROUP_NAME  = 3;

        ArrayList<Group> groups = CSVHandler.getDetectedGroups();

        double remainingRows = NUMBER_OF_ROWS_IN_PAGE - ROWS_OF_MAIN_HEADER;

        int questionIndex= 0 ;

        for (int groupIndex = 0; groupIndex < groups.size() ; groupIndex++) {

            int groupTableSize =groupsTables.get(groupIndex).get(0).size();

            remainingRows-= groupTableSize ;

            if(groupIndex>0) {
                doc.select("div.wrapper").last().after(groupNameHtml);
                Element lastGroupNameElement = doc.select("div.line").last();
                lastGroupNameElement.select("span.group-title").last().text(groups.get(groupIndex).getCleanedName());
                doc.select("div.line").last().after(wrapperHtml);

                remainingRows -= (ROWS_OF_TABLE_HEADER+ROWS_OF_GROUP_NAME);

                if (remainingRows < 0) {
                    lastGroupNameElement.before(pageBreakHtml);
                    remainingRows = NUMBER_OF_ROWS_IN_PAGE - ROWS_OF_GROUP_NAME - ROWS_OF_TABLE_HEADER ;
                }
            }
            ArrayList<ArrayList<ArrayList<String>>> tables = groupsTables.get(groupIndex);


            for (int tableIndex = 0; tableIndex < tables.size(); tableIndex++) {
                ArrayList<ArrayList<String>> table = tables.get(tableIndex);
                String spanClass = "second";
                if (tableIndex % 2 == 0) {
                    spanClass = "first";
                    if (tableIndex != 0) {
                        remainingRows -= (table.size() + ROWS_OF_TABLE_HEADER);
                        doc.select("div.wrapper").last().after(wrapperHtml);
                        if (remainingRows < 0) {
                            doc.select("div.wrapper").last().before(pageBreakHtml);
                            remainingRows = NUMBER_OF_ROWS_IN_PAGE -(table.size() + ROWS_OF_TABLE_HEADER);
                        }

                    }else { //for first table
                        doc.select("span.group-title").last().text(groups.get(groupIndex).getCleanedName());
                    }
                }
                String questionName = Statistics.getQuestionNames().get(questionIndex);
                ArrayList<ArrayList<String>> modifiedTable = Utils.cloneTable(table);

                for (ArrayList<String> tableRow : modifiedTable) {

                    String barWidth  = tableRow.get(tableRow.size() - 2);
                    String barClass = tableRow.get(tableRow.size() - 1);
                    String divHtml = "<div class='emptyBar'> \n <div class='"+barClass+"' style='width:" + barWidth + "%'></div>\n</div>";
                    tableRow.set( 3 , divHtml);
                }
                String tableRowsHtml = createRowsHtml(modifiedTable, ";gray-row", " ");
                Element nextTableToFillElement = doc.select("span." + spanClass).last();
                nextTableToFillElement.select("tr.header-row").after(tableRowsHtml);
                nextTableToFillElement.select("div.tableTitle").last().text(questionName);

                questionIndex++ ;
            }
            // if tables ended with odd number remove the last table
            if(groups.get(groupIndex).getqCount()%2==1)
                doc.select("span.second").last().remove() ;
        }

        doc.select("div.wrapper").last().attr("style" , "margin-bottom:0px") ;

        return doc ;
//        writeHtmlFile(pdfHtmlPath , doc);
    }
    @Override
    public void generateHtmlReport() throws IOException {
        Document doc = generatePdfHtml(groupsStatsTables) ;
        doc.select("div#footer").remove() ;
//        changeImgPath(doc, imagesFolderFullPath);
//        changeLegendImgPath(doc , imagesFolderFullPath);
        writeHtmlFile(outputFormatsFolderPaths[ReportsHandler.HTML]+outputFileName+".html" , doc);
    }

    @Override
    public void generatePdfReport() throws IOException, DocumentException {

        Document doc  = generatePdfHtml(groupsStatsTables);
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath, outputFormatsFolderPaths[ReportsHandler.PDF]+outputFileName+".pdf");
    }

    @Override
    public void generateTxtReport() {

        final int CHP = 2 ;

        ArrayList<String>headers= getHeaders() ;
//        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> printablegroupsStatsTables = getProcessedTables(ReportsHandler.TXT) ;
        String outputTxt = "" ;

        int pageWidth = TxtUtils.calcTableWidth(groupsStatsTables.get(0).get(0) , CHP);


        String txtTitle = TxtUtils.generateTitleLine(reportTitle,
                pageWidth,2) ;

        outputTxt+= txtTitle ;
        ArrayList<String> txtTables = new ArrayList<>() ;

        int questionIndex = 0 ;
        for(int groupIndex = 0  ;groupIndex <CSVHandler.getDetectedGroups().size() ; groupIndex++) {

            ArrayList<ArrayList<ArrayList<String>>> groupTables = Utils.clone3D(groupsStatsTables.get(groupIndex));
            for(int tableIndex = 0; tableIndex < groupTables.size() ; tableIndex++) {
                ArrayList<ArrayList<String>> table = groupTables.get(tableIndex) ;
                table = Utils.removeTableCol(table, 3) ;
                table.add(0,headers) ;
                String questionName = Statistics.getQuestionNames().get(questionIndex++);
                txtTables.add( TxtUtils.generateTxtTableAlignCenter(table,questionName ,CHP , false));
            }
        }

        outputTxt+= TxtUtils.stackTablesV(txtTables , 2) ;
        TxtUtils.writeTxtToFile(outputTxt , outputFormatsFolderPaths[ReportsHandler.TXT]+outputFileName+".txt");
    }



    private void editRowForPrintablePdf (ArrayList<String> tableRow) {
//        String rowClass = tableRow.get(tableRow.size()-1) ; // get last element
        int numberOfSolvers = Integer.valueOf(tableRow.get(1));
//        String addedImgName;
//        if(rowClass.equals("greenBar"))
//            addedImgName = "correct" ;
//        else if(numberOfSolvers == 0)
//            addedImgName = "nonDistractor";
//        else if(rowClass.equals("distBar"))
//            addedImgName = "distractor";
//        else {
//            addedImgName= null ;
//        }
//        String addedData = addedImgName!=null?
//                "<img src='"+addedImgName+".png' height='15px' class='type-img'> </img>":" " ;
//
////                                            tableRow.set(tableRow.size()-1 , "printable-bar");

        if(numberOfSolvers!=0)
            tableRow.set(tableRow.size()-1 , "printable-bar");
        else
            tableRow.set(tableRow.size()-1 , "");

//        tableRow.add(3 , addedData) ;

    }

    private void editRowForWord (ArrayList<String> tableRow) {
//        String infoImageHtml = tableRow.get(3);
//        if(!infoImageHtml.equals(" ")) {
//            int endIndex = infoImageHtml.indexOf(".png'") + 4;
//            String imgName = infoImageHtml.substring(10, endIndex) ;
//            String imgFullPath = workSpacePath + imgName ;
//            String imgEndcoding = "<<img,10,10>>"+ imgFullPath ;
//            tableRow.set(3, imgEndcoding);
//        }
        String percent = tableRow.get(2) ;
//        String color = tableRow.get(4);
//        color = color.substring(0 , color.length()-3) ;
        percent = percent.substring(0,percent.length()-2) ;
        String barImgFullPath = resourcesPath+"RectImages\\Report5\\green\\"+percent+".png" ;
        String imgEncoding = "<<img,70,10>>" + barImgFullPath ;
        tableRow.set(3,imgEncoding) ;
    }
    private  ArrayList<ArrayList<ArrayList<ArrayList<String>>>> getProcessedTables (int type){
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> printablegroupsStatsTables = new ArrayList<>();
        for(ArrayList<ArrayList<ArrayList<String>>> tables: groupsStatsTables) {
            ArrayList<ArrayList<ArrayList<String>>> clonedTables  = Utils.clone3D(tables) ;
            for ( ArrayList<ArrayList<String>> table :clonedTables ) {

                for(  ArrayList<String> tableRow :table){
                    if(type == ReportsHandler.PRINTABLE_PDF)
                        editRowForPrintablePdf(tableRow);
                    else
                        editRowForWord(tableRow);
                }
            }
            printablegroupsStatsTables.add(clonedTables);
        }
        return  printablegroupsStatsTables ;
    }

    @Override
    public void generatePrintablePdfReport() throws IOException, DocumentException {

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> printablegroupsStatsTables = getProcessedTables(ReportsHandler.PRINTABLE_PDF);
        Document doc = generatePdfHtml(printablegroupsStatsTables) ;
//        doc.select("th.percent").after("<th>  </th>") ;
        //change border color of empty bar
        doc.select("div.emptyBar").attr("style" , "border-color: #999999") ;

        styleTitlePrintable(doc);

        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath , outputFormatsFolderPaths[ReportsHandler.PRINTABLE_PDF]+outputFileName+".pdf");
    }

    private ArrayList<String> getHeaders (){
        ArrayList<String> headers = new ArrayList<>( );
        headers.add("Response") ;headers.add("Count") ; headers.add("Percent");;

        return headers ;
    }


    private String generateCharSeparatedValuesString(char separator) {
        ArrayList<String> headers = getHeaders() ;

//        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> printablegroupsStatsTables = getProcessedTables(ReportsHandler.TXT) ;
        String outputCsv = "" ;

        int pageWidth = CsvUtils.calcTableWidth(groupsStatsTables.get(0).get(0));

        String csvReportTitle = reportTitle  ;

        String txtTitle = CsvUtils.generateTitleLine(csvReportTitle,separator,
                pageWidth,2) ;

        outputCsv+= txtTitle ;
        ArrayList<String> formTxtTables = new ArrayList<>() ;

        int questionIndex = 0 ;
        for(int groupIndex = 0  ;groupIndex <CSVHandler.getDetectedGroups().size() ; groupIndex++) {

            ArrayList<ArrayList<ArrayList<String>>> groupTables = Utils.clone3D(groupsStatsTables.get(groupIndex));
            for(int tableIndex = 0; tableIndex < groupTables.size() ; tableIndex++) {
                ArrayList<ArrayList<String>> table = Utils.cloneTable(groupTables.get(tableIndex)) ;
                table = Utils.removeTableCol(table, 3) ;
                table.add(0,headers) ;
                String questionName = Statistics.getQuestionNames().get(questionIndex++);
                formTxtTables.add( CsvUtils.generateTable(table,separator,questionName));
            }
        }
        outputCsv+= CsvUtils.stackTablesV(formTxtTables , 2) ;
        return outputCsv;
    }
    @Override
    public void generateCsvReport() {

        String outputCsv = generateCharSeparatedValuesString(',') ;

        CsvUtils.writeCsvToFile(outputCsv , outputFormatsFolderPaths[ReportsHandler.CSV]+outputFileName+".csv");

    }

    @Override
    public void generateTsvReprot() {

        String outputCsv = generateCharSeparatedValuesString('\t') ;

        CsvUtils.writeCsvToFile(outputCsv , outputFormatsFolderPaths[ReportsHandler.TSV]+outputFileName+".tsv");

    }
    ArrayList<ArrayList<String>> getTableWithHeaders(ArrayList<ArrayList<String>> table) {
        ArrayList<String> headers = getHeaders() ;
//        headers.set(headers.size()-1 , "") ;
        headers.add("") ;

        ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(table) ;
        tableWithHeaders.add(0, headers) ;
        return tableWithHeaders ;
    }

    private void addWordLegend(XWPFDocument document) throws IOException, InvalidFormatException {
        ArrayList<ArrayList<String>> legends = new ArrayList<>();
        ArrayList<String> legend = new ArrayList<>( );

        legend.add(workSpacePath+"distractorColored.png") ; //legend img path
        legend.add("Distractors*") ; //legend txt
        legends.add(legend);

        legend = new ArrayList<>();
        legend.add(workSpacePath+"nonDistractorColored.png") ; //legend img path
        legend.add("Non Distractors") ; //legend txt
        legends.add(legend);

        legend = new ArrayList<>();
        legend.add(workSpacePath+"correctColored.png") ; //legend img path
        legend.add("Correct Answer") ; //legend txt
        legends.add(legend) ;


        WordUtils.addLegend(document , legends);
    }
    @Override
    public void generateWordReport() throws IOException, InvalidFormatException {

        final int LINE_ROWS = 4 ;
        final int TITLE_ROWS = 5 ;
        final int WRAPPER_TABLE_ROWS = 5 ;
        final int BLANK_PAGE_ROWS = 34 ;
        final int TABLE_SPACING_ROWS = 3 ;
//        final int FIRST_PAGE_ROWS = BLANK_PAGE_ROWS - TITLE_ROWS ;

        XWPFDocument document = WordUtils.createDocument((int)(WordUtils.inch * 0.9)) ;

        double tableWidth = WordUtils.pageWidth * 0.47 ;

        WordUtils.createWordFooter(document); ;

        ArrayList<Group>groups = CSVHandler.getDetectedGroups() ;
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> groupsProcessedTables = getProcessedTables(ReportsHandler.WORD) ;
        int remainingRows = 0 ;
        remainingRows = BLANK_PAGE_ROWS ;

        WordUtils.addTitle(document, reportTitle ,4 );

        remainingRows-= TITLE_ROWS ;

        for( int groupIndex = 0 ;groupIndex < groupsProcessedTables.size() ; groupIndex++ ) {
            ArrayList<ArrayList<ArrayList<String>>> groupTables = groupsProcessedTables.get(groupIndex);

            Group group = groups.get(groupIndex);


            if(remainingRows > LINE_ROWS + WRAPPER_TABLE_ROWS +groupTables.get(0).size()-TABLE_SPACING_ROWS) {
                WordUtils.addHeaderLine(document, group.getCleanedName());
            }else
            {
                if(remainingRows>0)
                    WordUtils.addPageBreak(document);
                remainingRows = BLANK_PAGE_ROWS ;
                WordUtils.addHeaderLine(document, group.getCleanedName());
            }

            remainingRows-= LINE_ROWS ;

            for( int tableIndex= 0 ; tableIndex <groupTables.size(); tableIndex +=2 ) {


                ArrayList<ArrayList<String>> table = groupTables.get(tableIndex);




                if(remainingRows < WRAPPER_TABLE_ROWS+table.size() -TABLE_SPACING_ROWS) {
                    if(remainingRows>0)
                        WordUtils.addPageBreak(document);
                    remainingRows = BLANK_PAGE_ROWS ;
                }

                remainingRows-= WRAPPER_TABLE_ROWS+table.size() ;

                ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(table);


                XWPFTable wrapperTable = document.createTable(1, 2);
                wrapperTable.setCellMargins(0, 0, 400, 0);
                XWPFTableRow tablerow = wrapperTable.getRow(0);
                XWPFTable leftTable = WordUtils.createTableInCell(tablerow.getCell(0), tableWithHeaders, WordUtils.TABLE_ALIGN_CENTER,
                        group.getName() + (tableIndex + 1), 13, false, tableWidth);

                WordUtils.setTableAlign(leftTable, ParagraphAlignment.LEFT);

                if (tableIndex + 1 < groupTables.size()) {
                    ArrayList<ArrayList<String>> table2 = groupTables.get(tableIndex + 1);

                    ArrayList<ArrayList<String>> tableWithHeaders2 = getTableWithHeaders(table2);

                    XWPFTable rightTable = WordUtils.createTableInCell(tablerow.getCell(1), tableWithHeaders2, WordUtils.TABLE_ALIGN_CENTER,
                            Utils.generatePattern(" ", 4) + group.getName() + (tableIndex + 2), 13, false, tableWidth);
                    WordUtils.setTableAlign(rightTable, ParagraphAlignment.RIGHT);
                }
                WordUtils.removeBorders(wrapperTable, false);
                WordUtils.setTableAlign(wrapperTable, ParagraphAlignment.CENTER);
                WordUtils.changeTableWidth(wrapperTable);

                document.createParagraph().createRun().addBreak();
            }
        }
        WordUtils.writeWordDocument(document , outputFormatsFolderPaths[ReportsHandler.WORD]+outputFileName+".docx");

    }

    @Override
    public void generateXlsReport() throws IOException {


        final int NUMBER_OF_COLS_BETWEEN_TABLES  = 3 ;
        final int NUMBER_OF_TABLE_COLS=3;
        int pageWidth = XlsUtils.PAGE_COl_PADDING * 2 + NUMBER_OF_TABLE_COLS   * 2+NUMBER_OF_COLS_BETWEEN_TABLES ;

        XlsUtils.createXls(pageWidth);

        XlsUtils.addTitle(reportTitle);

        ArrayList<Group>groups = CSVHandler.getDetectedGroups() ;
        int questionIndex= 0 ;

        for( int groupIndex = 0 ;groupIndex < groups.size() ; groupIndex++ ) {
            ArrayList<ArrayList<ArrayList<String>>> groupTables = groupsStatsTables.get(groupIndex);

            Group group = groups.get(groupIndex);

            XlsUtils.addHeaderLine(group.getCleanedName());

            for (int tableIndex = 0; tableIndex < groupTables.size(); tableIndex += 2) {
                ArrayList<ArrayList<String>> table = groupTables.get(tableIndex);

                ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(table);

                tableWithHeaders = Utils.removeTableCol(tableWithHeaders , 3);

                int linesAddedAfterLeftTable =0 ;
                if(tableIndex + 1 == groupTables.size())
                    linesAddedAfterLeftTable = XlsUtils.DEFAULT_NUMBER_OF_LINES_AFTER_TABLE;

                XlsUtils.addTableAlignCenter(tableWithHeaders,XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX,
                        Statistics.getQuestionNames().get(questionIndex),linesAddedAfterLeftTable);

                if (tableIndex + 1 < groupTables.size()) {
                    questionIndex++ ;
                    ArrayList<ArrayList<String>> table2 = groupTables.get(tableIndex + 1);

                    ArrayList<ArrayList<String>> tableWithHeaders2 = getTableWithHeaders(table2);
                    tableWithHeaders2 = Utils.removeTableCol(tableWithHeaders2 , 3);
                    XlsUtils.addTableAlignCenter(tableWithHeaders2,XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX+
                                    NUMBER_OF_TABLE_COLS+NUMBER_OF_COLS_BETWEEN_TABLES, Statistics.getQuestionNames().get(questionIndex),
                                    XlsUtils.DEFAULT_NUMBER_OF_LINES_AFTER_TABLE);
                }

                questionIndex ++ ;
            }
        }
        XlsUtils.postProcessSheet();
        int imgLabelWidth = 1000 ;
        XlsUtils.sheet.setColumnWidth(XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX+ NUMBER_OF_TABLE_COLS, imgLabelWidth);
        XlsUtils.sheet.setColumnWidth(XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX+ NUMBER_OF_TABLE_COLS+NUMBER_OF_COLS_BETWEEN_TABLES-1
                , imgLabelWidth);
        XlsUtils.writeXlsFile(outputFormatsFolderPaths[ReportsHandler.XLS]+outputFileName+".xls" , false );
    }

    @Override
    public void init() {
        Statistics.setQuestionNames(CSVHandler.getDetectedQHeaders());
        groupsStatsTables = new ArrayList<>();
        groupsStatsTables=Statistics.report9Stats() ;
    }
}
