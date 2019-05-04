package Jowil.Reports;

import Jowil.Reports.Utils.CsvUtils;
import Jowil.Reports.Utils.TxtUtils;
import Jowil.Reports.Utils.WordUtils;
import Jowil.Reports.Utils.XlsUtils;
import Jowil.Statistics;
import Jowil.Utils;
import com.lowagie.text.DocumentException;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import static Jowil.Reports.Utils.XlsUtils.addPictureToCell;

public class Report6 extends Report {
    String imgsDirectoryFullPath  ;
    String imgName = "DifficultyHistogram" ;
    ArrayList<ArrayList<ArrayList<String>>>  formsStatsTables ;

    volatile boolean chartsDone = false ;

    public Report6(){
     constructor();
    }

    public Report6 (String resoursesPath){
        super(resoursesPath) ;
        constructor();
    }
    private void constructor() {
        reportTitle = "Groups Insights Report" ;
        workSpacePath = reportsPath + "report6\\" ;
        templatePath = workSpacePath + "report6Template.html";
        pdfHtmlPath = workSpacePath+outputFileName+".html" ;
        imgsDirectoryFullPath =  workSpacePath ;
        while (!chartsDone) ;
    }


    /**
     * this function is called before the constructor
     * get the statistical calculations for statistics calss
     * generate the bar chart of the report
     */
    @Override
    public void init(){
        formsStatsTables = Statistics.report6Stats() ;

            Platform.runLater(new Runnable() {
                @Override
                public void run() {
                    try {
                        generateReport6Chart();
                        chartsDone = true ;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });
    }

    /**
     * generate a bar chart for reprot 6 that represent the group hardness, and stores it in image.
     * @throws IOException if it couldn't store the img in the spicified path
     */
    private void generateReport6Chart( ) throws IOException {
        for(int formIndex = 0 ; formIndex < formsStatsTables.size() ; formIndex++) {
            ArrayList<ArrayList<String>> statsTable = formsStatsTables.get(formIndex);

            ArrayList<ArrayList<String>> statsTableTrans = Utils.transposeStringList(statsTable);

            double[] hardness = statsTableTrans.get(6).stream().mapToDouble(d -> Double.valueOf(d)).toArray();
            ArrayList<String> groupNames = statsTableTrans.get(0) ;

            Stage stage = new Stage();
            stage.setTitle("Student Grades");
            final CategoryAxis xAxis = new CategoryAxis();
            final NumberAxis yAxis = new NumberAxis(0 , 10 , 1);
            final BarChart<String, Number> bc =
                    new BarChart<String, Number>(xAxis, yAxis);
            bc.setTitle("Group Difficulty");
            xAxis.setLabel("Group");
            yAxis.setLabel("Difficulty");

            XYChart.Series series1 = new XYChart.Series();
            bc.setLegendVisible(false);
            double max = 0;
            double min = 1000;
            int minIndex = 0;
            int maxIndex = 0;
            for (int groupIndex = 0; groupIndex < groupNames.size(); groupIndex++) {

                double groupHardness = hardness[groupIndex];
                if (groupHardness > max) {
                    max = groupHardness;
                    maxIndex = groupIndex;
                }
                if (groupHardness < min) {
                    min = groupHardness;
                    minIndex = groupIndex;
                }

                series1.getData().add(new XYChart.Data(groupNames.get(groupIndex), groupHardness));
            }
            bc.getData().add(series1);

            for (int groupIndex = 0; groupIndex < groupNames.size(); groupIndex++) {
                String addedClass = "normal";
                if (groupIndex == maxIndex)
                    addedClass = "hardest";
                if (groupIndex == minIndex)
                    addedClass = "easiest";
                Node n = bc.lookup(".data" + groupIndex + ".chart-bar");
                n.getStyleClass().add(addedClass);
            }


            bc.setAnimated(false);
            Scene scene = new Scene(bc, 800, 600);

            scene.getStylesheets().add("data/reports/report6/style.css");
            bc.applyCss();
            bc.layout();
            stage.setScene(scene);

            WritableImage snapShot = bc.snapshot(new SnapshotParameters(), null);
            ImageIO.write(SwingFXUtils.fromFXImage(snapShot, null), "png",
                    new File(workSpacePath + imgName + (formIndex+1) + ".png"));
        }
    }

    /**
     * generate the html document for the report. This html is then manipulated in different ways to be used in many formats
     * @return Document of the generated html
     * @throws IOException if it couldn't find the html template that it start with
     */

    private Document generatePdfHtml() throws IOException {
        final int MAX_NUMBER_OF_1PAGE_ROWS = 7;
        File file = new File(templatePath);
        Document doc = Jsoup.parse(file, "UTF-8");

        updateTemplateFooter(doc); // updates the date of the footer to the current date


        String templateBodyHtml = doc.select("div#template").html() ;
        final String pageBreakHtml= "<div class='page-break'></div>\n" ;


        for(int formIndex = 0 ; formIndex < formsStatsTables.size() ; formIndex++) {

            // append the template again for the new form
            if(formIndex>0) {
                doc.select("img").last().after(pageBreakHtml);
                doc.select("div.page-break").last().after(templateBodyHtml);
                doc.select("div.divTitle").addClass("second-page-header");
                doc.select("div.divTitle").last().text( reportTitle +": Form " + (formIndex+1) );
            } else if(Statistics.getNumberOfForms()>1)
                doc.select("div.divTitle").last().text(reportTitle +": Form " + (formIndex+1));

            ArrayList<ArrayList<String>> statsTable = formsStatsTables.get(formIndex);
            String tableRowsHtml = createRowsHtml(statsTable, ";grayRow", "tg-l711");

            doc.select("tr.headerRow").last().after(tableRowsHtml);

            // check if img need to be put in a new page
            if (statsTable.size() > MAX_NUMBER_OF_1PAGE_ROWS)
                doc.select("img").addClass("new-page-img");

            doc.select("img").last().attr("src" , imgName+(formIndex+1)+".png");
        }
        return doc  ;
    }



    @Override
    public void generateHtmlReport() throws IOException {
        Document doc = generatePdfHtml() ;
        doc.select("div#footer").remove() ;
        changeImgPath(doc , imgsDirectoryFullPath);
        doc.select("img").attr("width" , "60%") ;
        writeHtmlFile(outputFormatsFolderPaths[ReportsHandler.HTML]+outputFileName+".html" , doc);
    }

    @Override
    public void generatePdfReport() throws IOException, DocumentException {

        Document doc = generatePdfHtml() ;
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath,outputFormatsFolderPaths[ReportsHandler.PDF]+outputFileName+".pdf");

    }

    private ArrayList<ArrayList<String>> getTableWithHeaders ( ArrayList<ArrayList<String>> table) {

        ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(table);
        ArrayList<String>tableHeaders = new ArrayList<>();
        tableHeaders.add("Name") ; tableHeaders.add("Hardest Question") ;
        tableHeaders.add("Easiest Question") ; tableHeaders.add("Correct Response %") ;
        tableHeaders.add("Questions With Distractors %"); tableHeaders.add("Point Biserial");
        tableHeaders.add("Difficulity (0-10)");

        tableWithHeaders.add(0,tableHeaders);
//        tableWithHeaders.add(1,secondHeaders) ;
        return  tableWithHeaders ;
    }
//
//    private ArrayList<ArrayList<String>> getTableWithHeadersWord ( ArrayList<ArrayList<String>> table) {
//
//        ArrayList<ArrayList<String>> tableWithHeaders = Utils.cloneTable(table);
//        ArrayList<String>tableHeaders = new ArrayList<>();
//        tableHeaders.add("Section Name") ; tableHeaders.add("Hardest Question") ;
//        tableHeaders.add("Easiest Question") ; tableHeaders.add("Average Correct Percentage") ;
//        tableHeaders.add("Percentage Of Questions with Distractors"); tableHeaders.add("Average Point Biserial");
//        tableHeaders.add("Difficulity (0-10)");
//
//        tableWithHeaders.add(0,tableHeaders);
//        return  tableWithHeaders ;
//    }
    @Override
    public void generateTxtReport() {

        int CHP = 3 ;

        String outputTxt = "" ;

        ArrayList<ArrayList<String>> temp = new ArrayList<>( );
        temp = getTableWithHeaders(temp ) ;
        int pageWidth = TxtUtils.calcTableWidth(temp , CHP) ;
        for(int formIndex=  0 ; formIndex<formsStatsTables.size() ; formIndex++) {

          ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(formsStatsTables.get(formIndex)) ;


            String txtReportTitle = reportTitle  ;
            if(Statistics.getNumberOfForms()>1)
                txtReportTitle = txtReportTitle +": Form " + (formIndex+1) ;

            if(formIndex > 0 )
                outputTxt+= TxtUtils.newLine+Utils.generatePattern("*" , pageWidth)+TxtUtils.newLine;

            String txtTitle = TxtUtils.generateTitleLine(txtReportTitle,
                    TxtUtils.calcTableWidth(tableWithHeaders, CHP), 2);

            String txtTable = TxtUtils.generateTxtTableAlignCenter(tableWithHeaders, "", CHP, false);

             outputTxt+= txtTitle + txtTable;
        }
        TxtUtils.writeTxtToFile(outputTxt , outputFormatsFolderPaths[ReportsHandler.TXT]+outputFileName+".txt");
    }

    @Override
    public void generatePrintablePdfReport() throws IOException, DocumentException {
        Document doc = generatePdfHtml() ;

        styleTitlePrintable(doc);
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath,outputFormatsFolderPaths[ReportsHandler.PRINTABLE_PDF]+outputFileName+".pdf");
    }

    private String generateCharSeparatedValuesString(char separator) {
        String outputCsv = "" ;

        ArrayList<ArrayList<String>> temp = new ArrayList<>( );
        temp = getTableWithHeaders(temp ) ;
        int pageWidth = CsvUtils.calcTableWidth(temp) ;
        for(int formIndex=  0 ; formIndex<formsStatsTables.size() ; formIndex++) {

            ArrayList<ArrayList<String>> tableWithHeaders = getTableWithHeaders(formsStatsTables.get(formIndex)) ;


            String csvReportTitle = reportTitle  ;
            if(Statistics.getNumberOfForms()>1)
                csvReportTitle = csvReportTitle +": Form " + (formIndex+1) ;

//            if(formIndex > 0 )
//                outputCsv+= CsvUtils.newLine+Utils.generatePattern("*" , pageWidth)+CsvUtils.newLine;

            String txtTitle = CsvUtils.generateTitleLine(csvReportTitle, separator ,
                    pageWidth , 2);

            String txtTable = CsvUtils.generateTable(tableWithHeaders, separator);

            outputCsv+= txtTitle + txtTable;
        }
        return  outputCsv ;
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
        XWPFDocument document = WordUtils.createDocument(WordUtils.LANDSCAPE_PAGE_WIDHT , WordUtils.LANDSCAPE_PAGE_HEIGHT);

        WordUtils.createWordFooter(document); ;


        for(int formIndex = 0 ; formIndex < formsStatsTables.size() ; formIndex++) {


            String title = reportTitle ;
            if( formsStatsTables.size() >1) {
                title =   title+": Form " + (formIndex+1);
            }
            if(formIndex>0)
                WordUtils.addPageBreak(document);


            WordUtils.addTitle(document , title );

            WordUtils.addTable(document, getTableWithHeaders(formsStatsTables.get(formIndex)));

            WordUtils.addImage(document , imgsDirectoryFullPath+imgName+(formIndex+1)+".png");

        }
        WordUtils.writeWordDocument(document , outputFormatsFolderPaths[ReportsHandler.WORD]+outputFileName+".docx");
    }

    @Override
    public void generateXlsReport() throws IOException {

        int pageWidth = 9;

        XlsUtils.createXls(pageWidth);


        for(int formIndex = 0 ; formIndex < formsStatsTables.size() ; formIndex++) {

            ArrayList<ArrayList<String>>  tableWithHeaders = getTableWithHeaders(formsStatsTables.get(formIndex));


            String title = reportTitle;
            if (formsStatsTables.size() > 1) {
                title = title + ": Form " + (formIndex + 1);
            }

            XlsUtils.addTitle(title, 3);


            XlsUtils.addTableAlignCenter(tableWithHeaders);

            int imgColShift = 2 ;
            addPictureToCell(imgsDirectoryFullPath+imgName+(formIndex+1)+".png", XlsUtils.lastRowIndex,
                    XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX+imgColShift, 3, 10 , XlsUtils.DEFAULT_NUMBER_OF_LINES_AFTER_TABLE);

        }
        XlsUtils.writeXlsFile(outputFormatsFolderPaths[ReportsHandler.XLS]+outputFileName+".xls" );



    }

}
