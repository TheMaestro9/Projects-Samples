package Jowil.Reports;

import Jowil.CSVHandler;
import Jowil.Group;
import Jowil.Reports.Utils.CsvUtils;
import Jowil.Reports.Utils.TxtUtils;
import Jowil.Reports.Utils.WordUtils;
import Jowil.Reports.Utils.XlsUtils;
import Jowil.Statistics;
import Jowil.Utils;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.*;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.VerticalAlignment;
import org.apache.poi.xwpf.usermodel.*;
import org.jsoup.nodes.Document;
import com.lowagie.text.DocumentException;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.CTPPr;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.CTTblPr;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.CTVMerge;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.STMerge;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

public class Report8 extends Report {

    ArrayList<ArrayList<ArrayList<Double>>> formsData ;
    volatile boolean chartsReady = false ;
    String imagesFullPath ;
    String imgName ="GradualityChart";

    public Report8(){
     constructor();
    }


    public Report8 (String resoursesPath){
        super(resoursesPath) ;
        constructor();
    }
    private void constructor() {
        workSpacePath = reportsPath + "report8\\" ;
        templatePath = workSpacePath + "report8Template.html";
        pdfHtmlPath = workSpacePath+outputFileName+".html" ;
        imagesFullPath =  workSpacePath  ;
        while (!chartsReady) ;
    }

    public Document generatePdfHtml () throws IOException {
        File file = new File(templatePath);
        Document doc = Jsoup.parse(file, "UTF-8");

        updateTemplateFooter(doc); // updates the date of the footer to the current date

        String templateBodyHtml = doc.select("div#template").html() ;
        String wrapperHtml = doc.select("div.wrapper").outerHtml() ;
        final String pageBreakHtml= "<div class='page-break'></div>\n" ;
        final String spaceHtml = "<div class='give-me-space'></div>\n" ;

        ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;

        for (int formIndex= 0 ; formIndex < formsData.size() ; formIndex++ ) {
            ArrayList<ArrayList<Double>> formGraphsData = formsData.get(formIndex);
            if(formIndex>0) {
                doc.select("div.wrapper").last().after(pageBreakHtml);
                doc.select("div.page-break").last().after(templateBodyHtml) ;
                doc.select("div.divTitle").last().text( reportTitle+ ": Form " +(1+formIndex) ) ;
            }
            else if( formsData.size()>1)
                doc.select("div.divTitle").last().text(reportTitle+ ": Form " +(1+formIndex)) ;

            for ( int graphIndex = 0; graphIndex < formGraphsData.size() ; graphIndex++) {
               if(graphIndex > 0) {
                   if(graphIndex%2==0) {
                       doc.select("div.wrapper").last().after(pageBreakHtml);
                       doc.select("div.page-break").last().after(spaceHtml);
                       doc.select("div.give-me-space").last().after(wrapperHtml) ;
                   }
                   else {
                       doc.select("div.wrapper").last().after(wrapperHtml);
                   }
                   doc.select("span.group-title").last().text(groups.get(graphIndex-1).getCleanedName()) ;
               }
               ArrayList<Double> graphData = formGraphsData.get(graphIndex);
               graphData = new ArrayList<>(graphData.subList(graphData.size() - 3, graphData.size()));
               Elements tableCells = doc.select("table.t").last().select("td.right-td");
               for (int i = 0; i < tableCells.size(); i++) {
                   String cellText = Utils.formatNumber( graphData.get(i) , 1);
                   tableCells.get(i).text(cellText);
               }
               String imgName = this.imgName + formIndex + graphIndex + ".png";
               doc.select("img").last().attr("src", imgName);
           }
        }
        return doc ;
    }

    @Override
    public void generateHtmlReport() throws IOException {
        Document doc = generatePdfHtml();
        doc.select("div#footer").remove() ;
        doc.select("div.give-me-space").remove() ;
        doc.select("div.line").attr("style" , "margin-bottom:60px");
        changeImgPath(doc, imagesFullPath);
        writeHtmlFile(outputFormatsFolderPaths[ReportsHandler.HTML]+outputFileName+".html" , doc);

    }



    public void generateReport8Chart () throws IOException {

        for(int formIndex = 0 ; formIndex < formsData.size() ; formIndex ++) {
            ArrayList<ArrayList<Double>> formGraphsData = formsData.get(formIndex);

            for (int graphIndex = 0; graphIndex < formGraphsData.size(); graphIndex++) {
                ArrayList<Double> graphData = formGraphsData.get(graphIndex);

                int numberOfQuestions = graphData.size()-3 ;
                Stage stage = new Stage() ;

//                stage.setTitle("Difficulty Graduality");

                final NumberAxis xAxis = new NumberAxis(1 , numberOfQuestions, Math.max(Math.round((double)numberOfQuestions/10) , 1));
                final NumberAxis yAxis = new NumberAxis(0 , 10 , 1);
//                yAxis.setLowerBound(0);
//                yAxis.setUpperBound(10);
                final LineChart<Number,Number> lc =
                        new LineChart<>(xAxis,yAxis);

                xAxis.setLabel("Question Number");
                yAxis.setLabel("Difficulty");

                lc.setLegendVisible(false);
                lc.setCreateSymbols(false);

                lc.setPrefSize(800 , 500);
                lc.setAnimated(false) ;

                XYChart.Series series1 = new XYChart.Series();

                for (int questionIndex = 0; questionIndex < graphData.size() - 3; questionIndex++) {
                    series1.getData().add(new XYChart.Data(questionIndex + 1, graphData.get(questionIndex)));
                }


                lc.getData().add(series1);

                if(graphData.get(graphData.size()-3)<0) {
                    Node n = lc.lookup(".chart-series-line");
                    n.getStyleClass().add("bad");
                }
                Scene scene = new Scene(lc);
                scene.getStylesheets().add("data/reports/report8/style.css");
                lc.applyCss();
                lc.layout();
                stage.setScene(scene);


//                scene.getStylesheets().add("reports/report8/style.css");
//                lc.applyCss();
//                lc.layout();
//                stage.setScene(scene);
                WritableImage snapShot = lc.snapshot(new SnapshotParameters(), null);
                String imgName = workSpacePath + this.imgName + formIndex + graphIndex + ".png";
                ImageIO.write(SwingFXUtils.fromFXImage(snapShot, null), "png", new File(imgName));
                lc.getData().removeAll() ;
            }
        }
    }

    @Override
    public void generatePdfReport() throws IOException, DocumentException {
        Document doc  = generatePdfHtml();
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath, outputFormatsFolderPaths[ReportsHandler.PDF]+outputFileName+".pdf");
    }

    public ArrayList<ArrayList<String>> getTableWithHeaders (ArrayList<Double> data) {
        ArrayList<ArrayList<String>> table = new ArrayList<>( );
        ArrayList<String> tableRow = new ArrayList<>( );

        tableRow.add("Slope [-1, 1]") ; tableRow.add(Utils.formatNumber( data.get(0) , 1 )) ;
        table.add(tableRow );
        tableRow = new ArrayList<>( );
        tableRow.add("Error [0, 1]") ; tableRow.add(Utils.formatNumber( data.get(1) , 1 )) ;
        table.add(tableRow );
        tableRow = new ArrayList<>( );
        tableRow.add("JDP [0, 10]") ; tableRow.add(Utils.formatNumber( data.get(2) , 1 )) ;
        table.add(tableRow) ;

        return table ;
    }

    private ArrayList<String> getTableTitles () {
        ArrayList<String> titles = new ArrayList<>( );
        ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
        titles.add("all Exam") ;
        for (Group g : groups)
            titles.add(g.getCleanedName() ) ;

        return titles ;
    }
    @Override
    public void generateTxtReport() {

        int CHP  = 3  ;
        ArrayList<String> tablesTitles = getTableTitles();
        String outputTxt=  "" ;
        ArrayList<Double> tempTable = formsData.get(0).get(0) ;
        int pageWidth = TxtUtils.calcTableWidth(getTableWithHeaders(
                new ArrayList<>(tempTable.subList(tempTable.size()-3 , tempTable.size()))),CHP) ;
        for ( int formIndex = 0 ; formIndex < formsData.size() ; formIndex++ ) {
            String form  = "" ;
            if(formsData.size()>1)
                form = ": Form " + (formIndex+1) ;
            if(formIndex > 0)
                outputTxt+= Utils.generatePattern("*" , pageWidth) + TxtUtils.newLine ;
            String txtTitle = TxtUtils.generateTitleLine( reportTitle+form, pageWidth , 3) ;
            outputTxt += txtTitle ;

            ArrayList<ArrayList<Double>> formGraphsData = formsData.get(formIndex);
            ArrayList<String> txtTables = new ArrayList<>( );
            for (int graphIndex = 0; graphIndex < formGraphsData.size() ; graphIndex ++ ) {
                ArrayList<Double> graphData = formGraphsData.get(graphIndex);
                ArrayList<ArrayList<String>> table = getTableWithHeaders(new ArrayList<>
                        (graphData.subList(graphData.size() - 3, graphData.size())));
                txtTables.add( TxtUtils.generateTxtTableAlignLR(table, tablesTitles.get(graphIndex), CHP));
            }
            outputTxt += TxtUtils.stackTablesV(txtTables , 2) ;
        }

        TxtUtils.writeTxtToFile(outputTxt , outputFormatsFolderPaths[ReportsHandler.TXT]+outputFileName+".txt");
    }

    @Override
    public void generatePrintablePdfReport() throws IOException, DocumentException {
        Document doc = generatePdfHtml() ;

        styleTitlePrintable(doc);
        writeHtmlFile(pdfHtmlPath , doc);
        generatePDF(pdfHtmlPath , outputFormatsFolderPaths[ReportsHandler.PRINTABLE_PDF]+outputFileName+".pdf");
    }

    private String generateCharSeparatedValuesString(char separator) {
        ArrayList<String> tablesTitles = getTableTitles();
        String outputCsv=  "" ;
        ArrayList<Double> tempTable = formsData.get(0).get(0) ;
        int pageWidth = CsvUtils.calcTableWidth(getTableWithHeaders(
                new ArrayList<>(tempTable.subList(tempTable.size()-3 , tempTable.size())))) ;
        for ( int formIndex = 0 ; formIndex < formsData.size() ; formIndex++ ) {
            String form  = "" ;
            if(formsData.size()>1)
                form = ": Form " + (formIndex+1) ;
            String txtTitle = CsvUtils.generateTitleLine(reportTitle+form , separator,  pageWidth , 3) ;
            outputCsv += txtTitle ;

            ArrayList<ArrayList<Double>> formGraphsData = formsData.get(formIndex);
            ArrayList<String> txtTables = new ArrayList<>( );
            for (int graphIndex = 0; graphIndex < formGraphsData.size() ; graphIndex ++ ) {
                ArrayList<Double> graphData = formGraphsData.get(graphIndex);
                ArrayList<ArrayList<String>> table = getTableWithHeaders(new ArrayList<>
                        (graphData.subList(graphData.size() - 3, graphData.size())));
                txtTables.add( CsvUtils.generateTable(table, separator,tablesTitles.get(graphIndex)));
            }
            outputCsv += CsvUtils.stackTablesV(txtTables , 2) ;
        }

        return outputCsv ;
    }
        @Override
    public void generateCsvReport() throws IOException {

        String outputCsv = generateCharSeparatedValuesString(',') ;
        CsvUtils.writeCsvToFile(outputCsv , outputFormatsFolderPaths[ReportsHandler.CSV]+outputFileName+".csv");

    }

    @Override
    public void generateTsvReprot() {
        String outputTsv = generateCharSeparatedValuesString('\t') ;
        CsvUtils.writeCsvToFile(outputTsv , outputFormatsFolderPaths[ReportsHandler.TSV]+outputFileName+".tsv");
    }

    @Override
    public void generateWordReport() throws IOException, InvalidFormatException {


        XWPFDocument document = WordUtils.createDocument((int)(WordUtils.inch * 0.9)); // create document with left and right margin = 0.9inch


        WordUtils.createWordFooter(document); ;

        ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
        for ( int formIndex = 0 ; formIndex <formsData.size() ; formIndex++ ) {
            ArrayList<ArrayList<Double>> formGraphsData = formsData.get(formIndex);
            String title = reportTitle ;
            if( formsData.size() >1) {
                title =title +": Form "  + (formIndex+1) ;
            }
            if(formIndex>0)
                WordUtils.addPageBreak(document);

            WordUtils.addTitle(document, title , 3);

            for (int graphIndex = 0; graphIndex < formGraphsData.size(); graphIndex++) {
                // add page break after two graphs
                if(graphIndex%2 == 0 && graphIndex>0)
                    WordUtils.addPageBreak(document);

                ArrayList<Double> graphData = formGraphsData.get(graphIndex);
                ArrayList<ArrayList<String>> table = getTableWithHeaders(graphData);

                String titleLine = "All Test" ;
                if(graphIndex>0)
                    titleLine = groups.get(graphIndex-1).getCleanedName();
                WordUtils.addHeaderLine(document,titleLine);

                XWPFTable wrapperTable = document.createTable(1, 3);
                wrapperTable.setCellMargins(0, 0, 400, 0);
                XWPFTableRow tablerow = wrapperTable.getRow(0);
                WordUtils.createTableInCell(tablerow.getCell(0), table, WordUtils.TABLE_ALIGN_LR, "", 10, true , WordUtils.pageWidth*.43);
                String imgFullPath = imagesFullPath + this.imgName + formIndex + graphIndex + ".png";
                XWPFRun run = tablerow.getCell(1).getParagraphArray(0).createRun();
                run.setColor("FFFFFF");
                run.setText("man");
                WordUtils.addImage(tablerow.getCell(2).getParagraphArray(0), imgFullPath, 250, 150);

                WordUtils.removeBorders(wrapperTable, false);
                WordUtils.setTableAlign(wrapperTable, ParagraphAlignment.CENTER);
                WordUtils.changeTableWidth(wrapperTable);

                document.createParagraph().createRun().addBreak();
            }
        }

        WordUtils.writeWordDocument(document, outputFormatsFolderPaths[ReportsHandler.WORD] + outputFileName + ".docx");

    }

    @Override
    public void generateXlsReport() throws IOException {

        final int IMG_COLS = 6 ;
        final int IMG_ROWS = 6 ;
        final int TABLE_COLS = 6 ;
        final int TABLE_IMG_COL_SEPARATION = 3 ;

        final int IMG_COL_START_INDEX = XlsUtils.DEFAULT_TABLE_COl_STARTING_INDEX +TABLE_COLS +TABLE_IMG_COL_SEPARATION ;
        int pageWidth = XlsUtils.PAGE_COl_PADDING*2+ TABLE_COLS + TABLE_IMG_COL_SEPARATION + IMG_COLS ;

        XlsUtils.createXls(pageWidth) ;
        ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
        for ( int formIndex = 0 ; formIndex <formsData.size() ; formIndex++ ) {
            ArrayList<ArrayList<Double>> formGraphsData = formsData.get(formIndex);
            String title = reportTitle ;
            if( formsData.size() >1) {
                title =title +": Form "  + (formIndex+1) ;
            }

            XlsUtils.addTitle(title);

            for (int graphIndex = 0; graphIndex < formGraphsData.size(); graphIndex++) {

                ArrayList<Double> graphData = formGraphsData.get(graphIndex);
                ArrayList<ArrayList<String>> table = getTableWithHeaders(graphData);

                String titleLine = "All Test" ;
                if(graphIndex>0)
                    titleLine = groups.get(graphIndex-1).getCleanedName();

                XlsUtils.addHeaderLine(titleLine);

                XlsUtils.addReport8Table( table , TABLE_COLS);
                String imgFullPath = imagesFullPath + this.imgName + formIndex + graphIndex + ".png";

                XlsUtils.addPictureToCell(imgFullPath ,XlsUtils.lastRowIndex,
                        IMG_COL_START_INDEX,IMG_COLS,IMG_ROWS,XlsUtils.DEFAULT_NUMBER_OF_LINES_AFTER_TABLE+1);

            }
        }
        XlsUtils.writeXlsFile(outputFormatsFolderPaths[ReportsHandler.XLS]+outputFileName+".xls"  );
    }

    @Override
    public void init() {

        formsData = Statistics.report8Stats();
        Platform.runLater(new Runnable() {
            @Override
            public void run() {
                try {
                    generateReport8Chart();
                    chartsReady = true ;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
    }
}
