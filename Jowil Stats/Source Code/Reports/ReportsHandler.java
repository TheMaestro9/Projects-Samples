package Jowil.Reports;

import Jowil.*;
import com.lowagie.text.DocumentException;
import com.lowagie.text.pdf.BaseFont;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
//import org.jfree.io.FileUtilities;

import org.xhtmlrenderer.pdf.ITextRenderer;

import javax.imageio.ImageIO;
import java.io.*;
import java.lang.reflect.Array;
import java.text.AttributedCharacterIterator;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ReportsHandler {

    public final static int PDF=0,WORD=1,HTML=2,PRINTABLE_PDF=3 ,TXT=4,XLS=5, CSV = 6 , TSV = 7  ;
    public final static String [] FORMATS_NAMES={"PDF","WORD","HTML","PRINTABLE PDF","TXT","XLS","CSV","TSV"};
    private final String reportsPath=  ".\\src\\main\\resources\\reports\\";
    public static boolean isTestMode;
    AtomicBoolean isStopReportsGeneration;
    static volatile boolean showPDFPagesProgress;

    public ReportsHandler(){
        this(false);
    }
    public ReportsHandler(boolean isTestMode){
        ReportsHandler.isTestMode=isTestMode;
        translateFormContentToArabic();
    }



    public  void setIsStopReportsGeneration(boolean isStopReportsGeneration) {
        this.isStopReportsGeneration=new AtomicBoolean(isStopReportsGeneration);
    }
    public static boolean isShowPDFPagesProgress() {
        return showPDFPagesProgress;
    }


    public void generatePDF(String inputHtmlPath, String outputPdfPath) throws IOException, com.lowagie.text.DocumentException {

       String resourcesPath= ".\\src\\main\\resources\\" ;

        String url = new File(inputHtmlPath).toURI().toURL().toString();
        System.out.println("URL: " + url);



        OutputStream out = new FileOutputStream(outputPdfPath);



        //Flying Saucer part
        ITextRenderer renderer = new ITextRenderer(true);

        renderer.getFontResolver().addFont(resourcesPath+"font\\NotoNaskhArabic-Regular.ttf", BaseFont.IDENTITY_H, BaseFont.NOT_EMBEDDED);
        renderer.setDocument(url);
        renderer.layout();
        renderer.createPDF(out);

        out.close();

//        PdfReader pdfReader = new PdfReader(outputPdfPath);
//        pdfReader.selectPages("1");
//
//        PdfStamper pdfStamper = new PdfStamper(pdfReader,
//                new FileOutputStream(outputPdfPath));
//
//        pdfStamper.close();


    }


    public void generateReports(ArrayList<Report>Reports , ArrayList<Integer> formats  ) throws IOException, DocumentException, InvalidFormatException {

        boolean isPDFExists= (formats.contains(PDF));

        for(Report report:Reports) {
            if( !isTestMode && isStopReportsGeneration.get() ) //user cancelled generation
                return;

            if (formats.contains(PDF)){
                if(!isTestMode) {
                    showPDFPagesProgress = true;
                    ReportProgressWindow.removeSpinner();
                }
                report.generatePdfReport();
            }


            if(formats.contains(HTML))
                report.generateHtmlReport();

            if (formats.contains(TXT))
                report.generateTxtReport();

            if(formats.contains(WORD))
                report.generateWordReport();

            if(formats.contains(XLS))
                report.generateXlsReport();

            if(formats.contains(PRINTABLE_PDF)){
                showPDFPagesProgress=false;
                report.generatePrintablePdfReport();
            }


            if(formats.contains(CSV))
                report.generateCsvReport();
            if(formats.contains(TSV))
                report.generateTsvReprot();

            System.out.println("Finished Generating " + report.getClass().getSimpleName());

            if(!isPDFExists) {
                if(!handleNoPDF())
                    return;
            }

            if(!isTestMode)
                ReportProgressWindow.incrementProgressCount();

        }


    }

    private boolean handleNoPDF() {


        if(isTestMode)
            return true;

        if(isStopReportsGeneration.get())
            return false; //stop report generation if interrupted without pdf
        ReportProgressWindow.removeSpinner();
        ReportProgressWindow.setReportProgress(1);
        try {
            Thread.sleep(200); //simulating filling effect without pdf
        } catch (InterruptedException e) {
            System.out.println("dkhal el hramy wna nayma");
            return false; //stop report generation if interrupted without pdf
        }
        ReportProgressWindow.setReportProgress(0.0);
        return true;
    }


    public  void generateReport5Chart(Stage stage , ArrayList<String> responsePercentagesWithClasses  , ArrayList<String> questionChoices ) throws IOException {

       stage.setTitle("Bar Chart Sample");
       final NumberAxis yAxis = new NumberAxis(0 , 100 ,10);
       final CategoryAxis xAxis = new CategoryAxis();
       final BarChart<String,Number> bc =
               new BarChart<String,Number>(xAxis,yAxis);
//       bc.setTitle("Country Summary");
       bc.setLegendVisible(false);
//       xAxis.setLabel("Percentage");
//       xAxis.setTickLabelRotation(0);
//       yAxis.setLabel("");
       XYChart.Series series1 = new XYChart.Series();
       for(int choiceIndex =0 ; choiceIndex<questionChoices.size() ; choiceIndex++) {
           double responsePercent = Double.valueOf(responsePercentagesWithClasses.get(choiceIndex).split(";")[0])  ;
           series1.getData().add(new XYChart.Data( questionChoices.get(choiceIndex),responsePercent));
       }
       bc.getData().add(series1);


       for(int choiceIndex =0 ; choiceIndex<questionChoices.size() ; choiceIndex++) {
           String responseClass = responsePercentagesWithClasses.get(choiceIndex).split(";")[1] ;
           Node n = bc.lookup(".data"+choiceIndex+".chart-bar");
           if(responseClass.length()<2)
               responseClass = "normal";
           System.out.println(responseClass);
           n.getStyleClass().add(responseClass);
       }

       bc.setAnimated(false);
       Scene scene  = new Scene(bc,450,350);

       scene.getStylesheets().add("reports/report5/style.css");
       bc.applyCss();
       bc.layout();
       stage.setScene(scene);
       stage.show();

       WritableImage snapShot = bc.snapshot(new SnapshotParameters() , null);
       ImageIO.write(SwingFXUtils.fromFXImage(snapShot, null), "png", new File(reportsPath+"report5\\Q1stats.png"));

   }

    private void translateFormContentToArabic() {
        if(!Controller.isIsTranslateFormContent())
            return;

        translateQuestionsChoices();

        translateAnswersArray(Statistics.getCorrectAnswers());
        translateAnswersArray(Statistics.getStudentAnswers());


        if(Statistics.isIsIdentifierNumeric()){
            for(int i=0;i<Statistics.getStudentIdentifier().size();i++)
                Statistics.getStudentIdentifier().set(i,Translator.englishToArabic(Statistics.getStudentIdentifier().get(i)));
        }

    }

    private void translateQuestionsChoices() {
        int qIndex=0;
        for(Group group :CSVHandler.getDetectedGroups()){
            for(int i=0;i<group.getqCount();i++){
                for(int j=0;j<Statistics.getQuestionsChoices().get(qIndex).size();j++){
                    String choice=Statistics.getQuestionsChoices().get(qIndex).get(j);
                    String translated=group.isBinaryResponse()?Translator.englishToArabicBinary(choice):Translator.englishToArabic(choice);
                    Statistics.getQuestionsChoices().get(qIndex).set(j,translated);
                }
                qIndex++;
            }
        }
    }

    private void translateAnswersArray(ArrayList<ArrayList<String>> array) {

        for(int i=0;i<array.size();i++) {
            int qIndex=0;
            for (Group group : CSVHandler.getDetectedGroups()) {
                for (int j = 0; j < group.getqCount(); j++) {
                    String answer=array.get(i).get(qIndex);
                    String translated=group.isBinaryResponse()?Translator.englishToArabicBinary(answer):Translator.englishToArabic(answer);
                    array.get(i).set(qIndex,translated);
                    qIndex++;
                }
            }
        }
    }

}
