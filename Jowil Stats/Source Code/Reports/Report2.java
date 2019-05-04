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
import com.lowagie.text.Paragraph;
import com.lowagie.text.Table;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.xwpf.usermodel.*;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.CTHMerge;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.CTVMerge;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.STMerge;

import javax.print.Doc;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class Report2 extends Report {

    private ArrayList<Map<String , String>> formsStatsMaps ;
    private ArrayList<ArrayList<ArrayList<ArrayList<String>>>> formsStatsTables ;
    public static final int MAX_ACCEPTABLE_CHOICES = 10 ;
    public static final int MAX_NON_DESTRACOTRS = 6 ;
    private String templateName  ;
    ArrayList<ArrayList<ArrayList<ArrayList<String>>>> formsStatsPrintableTables ;

    final String lineHtml = "<div class=\"line\">\n" +
            "            <span class=\"group-title\">\n" +
            "                title\n" +
            "            </span>\n" +
            "        </div>" ;


    public Report2(){
        constructor();
    }

    public Report2 (String resoursesPath){
        super(resoursesPath) ;
        constructor();
    }
    private void constructor() {
        reportTitle = "Test Summary Report" ;
        workSpacePath = reportsPath + "report2\\" ;
        templatePath = workSpacePath + templateName ;
        pdfHtmlPath = workSpacePath+outputFileName+".html" ;
    }


    private Document generatePdfHtml(boolean isPrintable) throws IOException {
            File file = new File(templatePath);
            Document doc = Jsoup.parse(file, "UTF-8");

            updateTemplateFooter(doc); // updates the date of the footer to the current date

            final int ROWS_IN_BLANK_PAGE = 29 ;
            final int ROWS_IN_FIRST_PAGE = 12 ;
            final int NUMBER_OF_ROWS_FOR_TABLE_HEADER = 4 ;

            final int MINIMUM_REMAINING_ROWS = 4 +NUMBER_OF_ROWS_FOR_TABLE_HEADER;

            final String pageBreakHtml= "<div class='page-break'></div>\n" ;

            String tableHtml = doc.select("table.t2").last().outerHtml() ;
            String templateBodyHtml = doc.select("div#template").html() ;

            for (int formIndex = 0; formIndex < Statistics.getNumberOfForms() ; formIndex++) {
                if(formIndex>0) {
                    doc.select("table").last().after(pageBreakHtml);
                    doc.select("div.page-break").last().after(templateBodyHtml) ;
                    doc.select("div.divTitle").addClass("second-page-header") ;
                    doc.select("div.divTitle").last().text(reportTitle + ": Form "+(formIndex+1) );
                }
                else if(Statistics.getNumberOfForms()>1)
                    doc.select("div.divTitle").last().text(reportTitle + ": Form "+(formIndex+1) );

                fillGeneralStatsReport2(doc , formsStatsMaps.get(formIndex));
                ArrayList<ArrayList<ArrayList<String>>> statsTables ;
                if(isPrintable)
                    statsTables = formsStatsPrintableTables.get(formIndex);
                else
                    statsTables = formsStatsTables.get(formIndex);


//            fillGeneralStatsReport2(doc, Statistics.report2GeneralStats(formIndex));
//            ArrayList<ArrayList<ArrayList<String>>> statsTables = Statistics.report2TableStats(formIndex);
                int questionIndex = 0;
                int remainingRows = ROWS_IN_FIRST_PAGE;
                ArrayList<Group> groups = CSVHandler.getDetectedGroups();
                int groupIndex = 0 ;
                for (ArrayList<ArrayList<String>> table : statsTables) {
                    //create new table unless its first time
                    if (questionIndex != 0) {

                        //check if page break is needed
                        if (remainingRows < MINIMUM_REMAINING_ROWS  +  NUMBER_OF_ROWS_FOR_TABLE_HEADER -1 ) {
                            doc.select("table").last().after(pageBreakHtml);
                            doc.select("div.page-break").last().after(lineHtml) ;
                            doc.select("span.group-title").last().text(groups.get(groupIndex).getCleanedName());
                            remainingRows = ROWS_IN_BLANK_PAGE - NUMBER_OF_ROWS_FOR_TABLE_HEADER;
                            doc.select("div.line").last().after(tableHtml);

                        } else {
                            remainingRows -= (NUMBER_OF_ROWS_FOR_TABLE_HEADER *2-1);
                            doc.select("table").last().after(lineHtml) ;
                            doc.select("span.group-title").last().text(groups.get(groupIndex).getCleanedName());
                            doc.select("div.line").last().after(tableHtml);
                        }
                    }else // fill first line with group name
                        doc.select("span.group-title").last().text(groups.get(groupIndex).getCleanedName());

                    fillResponseFreqHeaders(doc, questionIndex);

                    //start and end indeces for questions to be shown in the page
                    int startIndex = 0;
                    int endIndex = (int) Utils.getNumberWithinLimits(table.size(), 0, remainingRows);
                    do {
                        //create html table
                        ArrayList<ArrayList<String>> pageTable = new ArrayList<ArrayList<String>>(table.subList(startIndex, endIndex));
                        String rowsHtml = createRowsHtml(pageTable, ";grayRow", "");
                        doc.select("tr.bottom-header-row").last().after(rowsHtml);

                        //update remaining rows counter
                        int numberOfInsertedRows = endIndex - startIndex;
                        remainingRows -= numberOfInsertedRows;
                        if (remainingRows < MINIMUM_REMAINING_ROWS && endIndex != table.size()) {
                            doc.select("table").last().after(pageBreakHtml);
                            //insert a new table in the new page
                            doc.select("div.page-break").last().after(tableHtml);
                            fillResponseFreqHeaders(doc, questionIndex);
                            remainingRows = ROWS_IN_BLANK_PAGE;
                        }
                        startIndex = endIndex;
                        endIndex = (int) Utils.getNumberWithinLimits(table.size(), 0, endIndex + remainingRows);
                    } while (startIndex != endIndex);
                    questionIndex += table.size();
                    groupIndex++ ;
                }
            }
            doc.select("table").last().attr("style" , "margin-bottom:0px") ;
            return doc ;
        }
        private void fillGeneralStatsReport2 (Document doc , Map<String , String> generalStatsMap) {
        doc.select("td.NumberOfStudents").last().text(generalStatsMap.get("Number Of Students")) ;
        doc.select("td.MaxPossibleScore").last().text(generalStatsMap.get("Maximum Possible Score")) ;

        //Basic Statistics
        doc.select("td.Mean").last().text(generalStatsMap.get("Mean Score")) ;
        doc.select("td.HighestScore").last().text(generalStatsMap.get("Highest Score")) ;
        doc.select("td.LowestScore").last().text(generalStatsMap.get("Lowest Score")) ;
        //Dispersion
        doc.select("td.StandardDeviation").last().text(generalStatsMap.get("Standard Deviation")) ;
        doc.select("td.ScoreRange").last().text(generalStatsMap.get("Range")) ;
        doc.select("td.Median").last().text(generalStatsMap.get("Median")) ;


        //Test Reliability
        doc.select("td.Kuder-RichardsonFormula20").last().text(generalStatsMap.get("Kuder-Richardson Formula 20")) ;

    }
    private void fillResponseFreqHeaders(Document doc , int questionIndex) {
        String questionChoicesHtml = "" ;
        ArrayList<String> questionChoices =  Statistics.getSpecificQuestionChoices(questionIndex) ;
        String addedClass = "" ;
        if(Utils.checkListContainArabic(questionChoices))
            addedClass =  "arabic-font";
        for(String qChoice: questionChoices )
            questionChoicesHtml+= "<th class='"+addedClass+"'>" +qChoice+ "</th>\n";
        doc.select("th.total").last().before(questionChoicesHtml);
        doc.select("th.responseFreq").last().attr("colspan" , String.valueOf(questionChoices.size())) ;

    }


    @Override
    public void generateHtmlReport() throws IOException {

        File file = new File(templatePath);
        Document doc = Jsoup.parse(file, "UTF-8");
        doc.select("div#footer").remove() ;

        String tableHtml = doc.select("table.t2").last().outerHtml() ;
        String templateBodyHtml = doc.select("div#template").html() ;

        for (int formIndex = 0; formIndex < Statistics.getNumberOfForms() ; formIndex++) {
            if(formIndex>0) {
                doc.select("table").last().after(templateBodyHtml);
                doc.select("div.divTitle").addClass("second-page-header") ;
                doc.select("div.divTitle").last().text( reportTitle+": Form "+(formIndex+1));
            }
            else if (Statistics.getNumberOfForms()> 1)
                doc.select("div.divTitle").last().text( reportTitle+": Form "+(formIndex+1));

            fillGeneralStatsReport2(doc , formsStatsMaps.get(formIndex));
            ArrayList<ArrayList<ArrayList<String>>> statsTables = formsStatsTables.get(formIndex);

//            fillGeneralStatsReport2(doc, Statistics.report2GeneralStats(formIndex));
//
//            ArrayList<ArrayList<ArrayList<String>>> statsTables = Statistics.report2TableStats(formIndex);
            int questionIndex = 0;

            ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
            int groupIndex = 0 ;
            for (ArrayList<ArrayList<String>> table : statsTables) {
                if(questionIndex!=0) {
                    doc.select("table").last().after(lineHtml) ;
                    doc.select("div.line").last().after(tableHtml);
                }
                doc.select("span.group-title").last().text(groups.get(groupIndex++).getCleanedName());
                fillResponseFreqHeaders(doc, questionIndex);
                String rowsHtml = createRowsHtml(table, ";grayRow", "");
                doc.select("tr.bottom-header-row").last().after(rowsHtml);
                questionIndex += table.size();
            }
        }
        writeHtmlFile(outputFormatsFolderPaths[ReportsHandler.HTML]+outputFileName+".html" , doc);

    }

    @Override
    public void generatePdfReport() throws IOException, DocumentException {
        Document doc = generatePdfHtml(false) ;
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath , outputFormatsFolderPaths[ReportsHandler.PDF]+outputFileName+".pdf");

    }

    private ArrayList<ArrayList<String>> processMap (Map<String , String> statsMap){
        ArrayList<ArrayList<String>> mapAsTable  = new ArrayList<>();
        ArrayList<String> tableRow = new ArrayList<>() ;

        tableRow.add("Number Of Students") ; tableRow.add(statsMap.get("Number Of Students")) ;
        tableRow.add("Mean") ; tableRow.add(statsMap.get("Mean Score")) ;
        tableRow.add("Lowest Score") ; tableRow.add(statsMap.get("Lowest Score")) ;

        mapAsTable.add(tableRow) ;
        tableRow = new ArrayList<>() ;

        tableRow.add("Maximum Possible Score") ; tableRow.add(statsMap.get("Maximum Possible Score")) ;
        tableRow.add("Median") ; tableRow.add(statsMap.get("Median"));
        tableRow.add("Highest Score") ; tableRow.add(statsMap.get("Highest Score")) ;

        mapAsTable.add(tableRow) ;
        tableRow = new ArrayList<>() ;

        tableRow.add("Score Range") ; tableRow.add(statsMap.get("Range")) ;
        tableRow.add("Standard Deviation") ; tableRow.add(statsMap.get("Standard Deviation"));
        tableRow.add("KR20") ; tableRow.add(statsMap.get("Kuder-Richardson Formula 20")) ;

        mapAsTable.add(tableRow) ;

        return mapAsTable ;
    }

    private ArrayList<String> getHeaders ( int questionIndex ) {

        ArrayList<String>tableHeaders = new ArrayList<>();
        tableHeaders.add("No.") ; tableHeaders.add("Question") ; tableHeaders.add("Correct Answer") ;

        ArrayList<String> questionChoices =  Statistics.getSpecificQuestionChoices(questionIndex) ;
        for(String qChoice: questionChoices )
            tableHeaders.add(qChoice) ;
        tableHeaders.add("Non Distractors") ;
        tableHeaders.add("Point Biserial") ; tableHeaders.add("Overall %") ; tableHeaders.add("Bottom 25%");
        tableHeaders.add("Top 25%") ;

        return tableHeaders ;
    }

    private int calcTxtPageWidth(int chpCenterTable  ,int chpLRTable )  {
        ArrayList<Integer> tabelsChp  = new ArrayList<>( );
        ArrayList<ArrayList<ArrayList<String>>> firstFormTables = new ArrayList<>();

        int questionIndex = 0 ;
        for(ArrayList<ArrayList<String>> table:formsStatsPrintableTables.get(0)) {
            ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(table);
            tableWithHeaders.add(0, getHeaders(questionIndex));
            tableWithHeaders = cleanTable(tableWithHeaders) ;
            firstFormTables.add(tableWithHeaders) ;
            tabelsChp.add(chpCenterTable) ;
            questionIndex+= tableWithHeaders.size() - 1 ;
        }
        firstFormTables.add(processMap( formsStatsMaps.get(0)) )  ;
        tabelsChp.add(chpLRTable) ;
        return  TxtUtils.calcPageWidth(firstFormTables , tabelsChp) ;
    }
    @Override
    public void generateTxtReport() {
        final int CHP_CENTER_TABLE = 3 ; // cell horizontal padding for center aligned tables
        final  int CHP_LR_TABLE = 5 ; // cell horizontal padding for Tables aligned Left Right
        final int PADDING_BETWEEN_TABLES = 2 ;

        String outputTxt="" ;


        int pageWidth = calcTxtPageWidth(CHP_CENTER_TABLE , CHP_LR_TABLE);

        for(int formIndex = 0 ; formIndex < Statistics.getNumberOfForms() ; formIndex++) {
            Map<String, String> statsMap = formsStatsMaps.get(formIndex);
            ArrayList<ArrayList<ArrayList<String>>> formsStatsTables = formsStatsPrintableTables.get(formIndex);



            ArrayList<ArrayList<String>> mapAsTable = processMap(statsMap);
            String generalStatsTxt = TxtUtils.generateTxtTableAlignLR(mapAsTable, "", CHP_LR_TABLE);

            String txtReportTitle = reportTitle  ;
            if(Statistics.getNumberOfForms()>1)
                txtReportTitle =  txtReportTitle + ": Form "+(formIndex+1)  ;

            String txtTitle = TxtUtils.generateTitleLine(txtReportTitle,
                    pageWidth,2) ;

            if(formIndex>0)
                outputTxt += Utils.generatePattern("#" , pageWidth)+TxtUtils.newLine;

            outputTxt+= txtTitle ;

            outputTxt+= generalStatsTxt  ;

            String legend = "* : Distractor"+TxtUtils.newLine ;

            outputTxt+= Utils.generatePattern(TxtUtils.newLine , 2 ) + legend ;

            ArrayList<String> txtTables = new ArrayList<>();


            int questionIndex = 0;

            for(int tableIndex = 0 ; tableIndex< formsStatsTables.size() ; tableIndex++ ) {
                ArrayList<ArrayList<String>> statsTable = cleanTable(formsStatsTables.get(tableIndex));
                ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(statsTable);
                tableWithHeaders.add(0, getHeaders(questionIndex));
                txtTables.add(TxtUtils.generateTxtTableAlignCenter(tableWithHeaders, "", CHP_CENTER_TABLE , false));
                questionIndex += statsTable.size() ;
             }

             outputTxt += TxtUtils.stackTablesV(txtTables, PADDING_BETWEEN_TABLES) ;
            }
//        String outputTxt = TxtUtils.stackTablesV(tables, PADDING_BETWEEN_TABLES) ;

        TxtUtils.writeTxtToFile(outputTxt , outputFormatsFolderPaths[ReportsHandler.TXT]+outputFileName+".txt");

    }

    @Override
    public void generatePrintablePdfReport() throws IOException, DocumentException {
        Document doc = generatePdfHtml(false) ;

        String printableLegendHtml = "<div class=\"wrapper\">\n" +
                "            <span class=\"bold\"> Correct Answer :</span>\n" +
                "            <span class=\"second \"> Bold</span>\n" +
                "        </div>\n" +
                "        <div class=\"wrapper\" style=\"margin-bottom: 10px\">\n" +
                "            <span class=\"bold\"> Distractor : </span>\n" +
                "            <span class=\"second\" > Bold  *  </span>\n" +
                "        </div>" ;
//        String correctAnswerHtml = "<th rowspan='2' >Correct <br> Answer</th>" ;
//        String nonDistractorHtml = "<th rowspan='2' >Non <br> Distractors</th>" ;
//
        doc.select("div.legend").html(printableLegendHtml) ;
//        doc.select("th.question").after(correctAnswerHtml) ;
//        doc.select("th.point-biserial").before(nonDistractorHtml) ;

        doc.select("td.green").removeClass("green") ;
        doc.select("td.red").removeClass("red");
        doc.select("td.gold").removeClass("gold") ;
        styleTitlePrintable(doc);
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath , outputFormatsFolderPaths[ReportsHandler.PRINTABLE_PDF]+outputFileName+".pdf");
    }

    private String generateCharSeparatedValuesString(char separator) {
        final int PADDING_BETWEEN_TABLES = 2 ;

        String outputCsv="" ;


        int pageWidth = CsvUtils.calcPageWidth(formsStatsPrintableTables.get(0)); // calc page width based on first form tables

        for(int formIndex = 0 ; formIndex < Statistics.getNumberOfForms() ; formIndex++) {
            Map<String, String> statsMap = formsStatsMaps.get(formIndex);
            ArrayList<ArrayList<ArrayList<String>>> formsStatsTables = formsStatsPrintableTables.get(formIndex);


            ArrayList<ArrayList<String>> mapAsTable = processMap(statsMap);
            String generalStatsTxt = CsvUtils.generateTable(mapAsTable, separator);

            String csvReportTitle = reportTitle  ;
            if(Statistics.getNumberOfForms()>1)
                csvReportTitle =  csvReportTitle +": Form "+(formIndex+1) ;

            String txtTitle = CsvUtils.generateTitleLine(csvReportTitle, separator,
                    pageWidth,2) ;


            outputCsv+= txtTitle ;

            outputCsv+= generalStatsTxt  ;

            String legend = "* : Distractor"+CsvUtils.NEW_LINE ;

            outputCsv+= Utils.generatePattern(TxtUtils.newLine , 2 ) + legend ;

            ArrayList<String> csvTables = new ArrayList<>();


            int questionIndex = 0;

            for(int tableIndex = 0 ; tableIndex< formsStatsTables.size() ; tableIndex++ ) {
                ArrayList<ArrayList<String>> statsTable = cleanTable(formsStatsTables.get(tableIndex));
                ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(statsTable);
                tableWithHeaders.add(0, getHeaders(questionIndex));
                csvTables.add(CsvUtils.generateTable(tableWithHeaders, separator));
                questionIndex += statsTable.size() ;
            }

            outputCsv += CsvUtils.stackTablesV(csvTables, PADDING_BETWEEN_TABLES) ;
        }

        return outputCsv ;
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
    private void addWordLegend(XWPFDocument document) throws IOException, InvalidFormatException {
        ArrayList<ArrayList<String>> legends = new ArrayList<>();
        ArrayList<String> legend = new ArrayList<>( );

        legend.add(workSpacePath+"redLegend.png") ; //legend img path
        legend.add("Distractors*") ; //legend txt
        legends.add(legend);

        legend = new ArrayList<>();
        legend.add(workSpacePath+"goldLegend.png") ; //legend img path
        legend.add("Non Distractors") ; //legend txt
        legends.add(legend);

        legend = new ArrayList<>();
        legend.add(workSpacePath+"greenLegend.png") ; //legend img path
        legend.add("Correct Answer") ; //legend txt
        legends.add(legend) ;


        WordUtils.addLegend(document , legends);
    }

    public void createWordHeader(XWPFDocument document , int questionIndex ) {
        ArrayList<String> headers = getHeaders(questionIndex);
        XWPFTable headerTable = document.createTable(2 , headers.size()) ;

        WordUtils.changeTableWidth(headerTable);
        int numberOfChoices = headers.size() - 8 ;
        CTVMerge vmerge = CTVMerge.Factory.newInstance();
        vmerge.setVal(STMerge.RESTART);

        CTHMerge hmerge = CTHMerge.Factory.newInstance();
        hmerge.setVal(STMerge.RESTART);

        CTVMerge vmerge2 = CTVMerge.Factory.newInstance();
        vmerge2.setVal(STMerge.CONTINUE);

        CTHMerge hmerge2 = CTHMerge.Factory.newInstance();
        hmerge2.setVal(STMerge.CONTINUE);

        String[] firstRowHeaders = {"Response Percentages" , "Correct Group Responses"} ;
        int firstRowIndex = 0  ;

        for(int rowIndex = 0 ; rowIndex< 2 ; rowIndex++) {
            for(int colIndex =0 ; colIndex<headers.size() ; colIndex++) {
                XWPFTableCell cell = headerTable.getRow(rowIndex).getCell(colIndex) ;
                XWPFParagraph headerCellPar = cell.getParagraphArray(0) ;
                headerCellPar.setSpacingAfter(0);
                headerCellPar.setAlignment(ParagraphAlignment.CENTER);
                XWPFRun headerRun = headerCellPar.createRun();
                headerRun.setBold(true);
                if (cell.getCTTc().getTcPr() == null)
                    cell.getCTTc().addNewTcPr();
                if(colIndex<3 || (colIndex > 2 +numberOfChoices &&  colIndex < headers.size()-3) ){
                    if(rowIndex ==0) {
                        cell.getCTTc().getTcPr().setVMerge(vmerge);
                        cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
                        headerRun.setText(headers.get(colIndex));
                    }
                    else
                        cell.getCTTc().getTcPr().setVMerge(vmerge2);
                }
                else if(rowIndex==0 && (colIndex==3 || colIndex == headers.size()-3)) {
                    cell.getCTTc().getTcPr().setHMerge(hmerge);
                    headerRun.setText(firstRowHeaders[firstRowIndex++]);
                }
                else if(rowIndex==0)
                    cell.getCTTc().getTcPr().setHMerge(hmerge2);
                else headerRun.setText(headers.get(colIndex));

            }
        }



    }
    @Override
    public void generateWordReport() throws IOException, InvalidFormatException {

        XWPFDocument document = WordUtils.createDocument(WordUtils.LANDSCAPE_PAGE_WIDHT , WordUtils.LANDSCAPE_PAGE_HEIGHT) ; // landscape size
//
        WordUtils.createWordFooter(document); ;
        for(int formIndex = 0 ; formIndex < formsStatsTables.size() ; formIndex++) {
            String title = reportTitle;
            if( formsStatsTables.size() >1) {
                title = title +": Form " + (formIndex+1);
            }
            if(formIndex>0)
                WordUtils.addPageBreak(document);

            WordUtils.addTitle(document , title);




            ArrayList<ArrayList<String>> mapTable = processMap(formsStatsMaps.get(formIndex)) ;
            ArrayList<ArrayList<ArrayList<String>>> statsTables = formsStatsTables.get(formIndex);

            WordUtils.addTable(document , mapTable , WordUtils.TABLE_ALIGN_LR , "" , 14);

            addWordLegend(document);

            int questionIndex = 0 ;
            for(int tableIndex = 0 ; tableIndex<statsTables.size() ; tableIndex++) {
                ArrayList<ArrayList<String>> table = statsTables.get(tableIndex);
//                ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(table) ;
//                tableWithHeaders.add(0, getHeaders(questionIndex)) ;
                createWordHeader(document , questionIndex );
                WordUtils.addTable(document , table , true , false);
                questionIndex+= table.size() ;
            }
        }
        WordUtils.writeWordDocument(document , outputFormatsFolderPaths[ReportsHandler.WORD]+outputFileName+".docx");

    }

    @Override
    public void generateXlsReport() throws IOException {



        int pageWidth = CsvUtils.calcPageWidth(formsStatsPrintableTables.get(0)) + 2; // calc page width based on first form tables
        XlsUtils.createXls(pageWidth);
        for(int formIndex = 0 ; formIndex < Statistics.getNumberOfForms() ; formIndex++) {
            Map<String, String> statsMap = formsStatsMaps.get(formIndex);
            ArrayList<ArrayList<ArrayList<String>>> formStatsTables = formsStatsTables.get(formIndex);


            ArrayList<ArrayList<String>> mapAsTable = processMap(statsMap);

            String xlsReportTitle = reportTitle  ;
            if(Statistics.getNumberOfForms()>1)
                xlsReportTitle =  xlsReportTitle +": Form "+(formIndex+1) ;

            XlsUtils.addTitle(xlsReportTitle,3);

            XlsUtils.addTableAlignLR(mapAsTable,"");

            XlsUtils.addPictureToCell(workSpacePath+ "legend.PNG",XlsUtils.lastRowIndex , XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX ,
                    1 , 3 , 1);

//            String legend = "* : Distractor"+CsvUtils.NEW_LINE ;
//
//            outputCsv+= Utils.generatePattern(TxtUtils.newLine , 2 ) + legend ;
//
//            ArrayList<String> csvTables = new ArrayList<>();


            int questionIndex = 0;

            ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
            for(int tableIndex = 0 ; tableIndex< formStatsTables.size() ; tableIndex++ ) {
                XlsUtils.addHeaderLine(groups.get(tableIndex).getCleanedName());
                ArrayList<ArrayList<String>> statsTable = formStatsTables.get(tableIndex);
                ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(statsTable);
                tableWithHeaders.add(0, getHeaders(questionIndex));
                XlsUtils.addTableAlignCenter(tableWithHeaders);
                questionIndex += statsTable.size() ;
            }

        }

        XlsUtils.writeXlsFile(outputFormatsFolderPaths[ReportsHandler.XLS]+outputFileName+".xls" );

    }

    @Override
    public void init()  {
        formsStatsMaps = new ArrayList<>() ;
        formsStatsTables = new ArrayList<>();
        formsStatsPrintableTables = new ArrayList<>();
        for (int formIndex = 0 ; formIndex < Statistics.getNumberOfForms() ; formIndex++) {
            formsStatsMaps.add(Statistics.report2GeneralStats(formIndex)) ;
            formsStatsTables.add(Statistics.report2TableStats(formIndex)) ;
            formsStatsPrintableTables.add(Statistics.report2PrintableStats(formsStatsTables.get(formIndex) , formIndex)) ;
        }

//        System.out.println("Iam in report 2 init man ");
        templateName = "report2Template.html" ;
        if(Statistics.getMaxNumOfChoices() >MAX_ACCEPTABLE_CHOICES)
            templateName = "report2LongTemplate.html" ;

        System.out.println("template name in init" + templateName);
    }
}
