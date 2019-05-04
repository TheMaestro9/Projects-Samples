package Jowil;

import Jowil.Reports.Report;
import Jowil.Reports.Report1;
import Jowil.Reports.ReportsHandler;
import Jowil.Reports.Utils.TxtUtils;
import Jowil.Reports.Utils.WordUtils;
import com.lowagie.text.DocumentException;


import javafx.scene.SubScene;
import javafx.scene.chart.StackedAreaChart;
import javafx.util.Pair;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation ;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.StatUtils.* ;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import java.awt.*;
import java.io.*;
import java.net.URLDecoder;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import static com.sun.javafx.scene.control.skin.Utils.getResource;
import static org.apache.commons.math3.stat.StatUtils.max;


public class Test {

    /**
     * function to get the slope and the average error of the trend for question hardness data sequence
     * @param hardness the ordered hardness of questions at index 0 question 1 ... etc
     * @return Pair with key = slope and value = error
     */

    public static Pair <Double,Double> getTrendDataSimple (ArrayList<Double> hardness)  {
        SimpleRegression regression = new SimpleRegression() ;

        double[] hardnessArray = hardness.stream().mapToDouble(d -> d).toArray();
        double maxHardness = max(hardnessArray);

        for(int i = 0 ; i < hardness.size() ; i ++) {
            regression.addData((double)i/(hardness.size()-1) , hardness.get(i)/maxHardness);
        }
        System.out.println("slope: " + regression.getSlope());
        System.out.println("intercept: "+ regression.getIntercept());
//        System.out.println("Mean Square Error: "+regression.getMeanSquareError());
//        System.out.println("R Square: "+regression.getRSquare());
//        System.out.println("R: "+regression.getR());
//        System.out.println("Slope Std Error: "+regression.getSlopeStdErr());
//        System.out.println("Sum Sqared Error: "+regression.getSumSquaredErrors());

        return new Pair<Double, Double>(regression.getSlope() , Math.sqrt(regression.getMeanSquareError()) * 1) ;
    }

    private static  double calcHarMean (double x , double y ) {
        return  2 * (x * y) / ( x + y) ;
    }

    private static double calcJowilParam (double slope , double error) {
        double slopeSign ;
        if(slope == 0)
            slopeSign = 1 ;
        else
            slopeSign = slope/Math.abs(slope) ;
        double harMean = calcHarMean(Math.abs(slope) , 1-Math.abs(error)) ;
        return (slopeSign* harMean + 1)*5;
    }

    public static Pair<Double , Double> getTrendData (ArrayList<Double> hardness) {
        double[] hardnessArray = hardness.stream().mapToDouble(d -> d).toArray();
        double [][] X = new double[hardnessArray.length][1] ;

        double maxHardness = max(hardnessArray);
        for (int i = 0 ; i< hardnessArray.length ; i++) { // x axis = 0 --> 1
            X[i][0] = (double)i / (double)(hardnessArray.length - 1);
            hardnessArray[i]/=maxHardness ;  // to make hardness 0 --> 1
        }
        OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
        ols.newSampleData(hardnessArray , X) ;
        RealMatrix coff = MatrixUtils.createColumnRealMatrix(ols.estimateRegressionParameters());
        double slope = coff.getColumnVector(0).getEntry(1);
        double Rss = ols.estimateRegressionStandardError();
        System.out.println("Resedual sum squared: "+ols.calculateResidualSumOfSquares());
//        ols.
        return new Pair<Double, Double>(slope , Rss) ;
        }

        public static char getReplacement (char char1 , char char2 ){
            int shiftedDiff = Math.abs(char1 - char2) ;
            if(shiftedDiff == 2)
                return 'b' ;
            else if(Math.min(char1 , char2)-'a' == 0)
                return 'c';
            else
                return 'a';
        }



    public static String getSerialNumber() {

        String dir = System.getProperty("user.dir");
        String drive = dir.split(":")[0] ;
        String result = "";
        try {
            File file = File.createTempFile("realhowto",".vbs");
            file.deleteOnExit();
            FileWriter fw = new java.io.FileWriter(file);

            String vbs = "Set objFSO = CreateObject(\"Scripting.FileSystemObject\")\n"
                    +"Set colDrives = objFSO.Drives\n"
                    +"Set objDrive = colDrives.item(\"" + drive + "\")\n"
                    +"Wscript.Echo objDrive.SerialNumber";  // see note
            fw.write(vbs);
            fw.close();
            Process p = Runtime.getRuntime().exec("cscript //NoLogo " + file.getPath());
            BufferedReader input =
                    new BufferedReader
                            (new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = input.readLine()) != null) {
                result += line;
            }
            input.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        return result.trim();
    }

    public static String getVolSerialNumber(String drive) {

        String result = "";
        try {

            Process p = Runtime.getRuntime().exec("cmd.exe /C vol");
            BufferedReader input =
                    new BufferedReader
                            (new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = input.readLine()) != null) {
                result += line;
            }
            input.close();
        }
        catch(Exception e){
        e.printStackTrace();
        }

        String  [] words=result.split(" ");

        return words[words.length-1].replace("-","");
    }

    static ArrayList<Character> allChars;
    public static void initAllChars(){

        allChars = new ArrayList<>( );

        for(int i = 48 ; i < 58 ; i++) {
            allChars.add((char)(i)) ; // add numbers
        }


        for(int i = 65 ; i < 91 ; i++) {
            allChars.add((char)(i)) ; // add numbers
        }
    }
    public static String getActivationKey(long serialNumber){

        Random gen = new Random(serialNumber) ;

        ArrayList<Character> shuffeledChars = (ArrayList)allChars.clone() ;
        Collections.shuffle((ArrayList)shuffeledChars,gen) ;
        String output = "";
        for (int i = 1; i < 17; i++) {
            output+=allChars.get(Math.abs(gen.nextInt())%36) ;
            if (i % 4 == 0 && i < 16)
                output += "-";
        }
        return output;
    }
    public static void main(String [] args) throws IOException, DocumentException {


//        initAllChars();
//
//        String volSerial = getVolSerialNumber("") ;
//        System.out.println("Serial Number: "+volSerial);
//        String activation = getActivationKey(Long.parseLong(volSerial , 16) ) ;
//
//        System.out.println("Activation Key: " + activation);

        System.out.println(Utils.checkStringEnglish("9.9")) ;
//        String s = "D+";
//        System.out.println(s.matches("\\w+"));
//        allChars.add('x');

//        initAllChars();
//        System.out.println(allChars.size());

//        long baseNumber = Long.valueOf(getSerialNumber()) ;
//        System.out.println(getSerialNumber("c"));
//        String output = getActivationKey(baseNumber) ;
//        System.out.println("the serial Number: " + baseNumber);
//        System.out.println("the result output: " + output);
//        String sn = getVolSerialNumber("C");

//        System.out.println(sn);

//        System.out.println(Integer.parseInt("cd" , 16));

//        File file = new File("E:\\work\\Jowil\\output folder test\\Jowil\\XLS Reports\\Report1 - Grades Distribution Report.xls");
//
//        //first check if Desktop is supported by Platform or not
//
//        if(!Desktop.isDesktopSupported()){
//            System.out.println("Desktop is not supported");
//            return;
//        }
//
//
//        Desktop desktop = Desktop.getDesktop();
//        if(file.exists()) desktop.open(file);

        //let's try to open PDF file
//        file = new File("/Users/pankaj/java.pdf");
//        if(file.exists()) desktop.open(file);

//        StringBuilder builder = new StringBuilder();
//        builder.append("hi Man") ;
//        builder.append("كيفك");
//        builder.append(" ") ;
//        builder.append("10");
//
//
//        System.out.println(h);
//        System.out .println(h);
//        System.out.println( builder.toString());
//        TxtUtils.writeTxtToFile(builder.toString() , "test.txt");


//        ArrayList<Double> hardness = new ArrayList<>( );
//        for(double i = 0 ; i < 40 ; i ++) {
////            hardness.add(0.0) ;
//            hardness.add(i / 4);
////            hardness.add(i-10);
//        }
//        for(int i = 0 ; i <10 ; i ++)
//            hardness.set(i , hardness.get(i) + 1) ;
////        hardness.add(0.0); hardness.add(3.0);
////        hardness.add(6.0); hardness.add(4.0); hardness.add(3.0); hardness.add(4.0) ;
////        hardness.add(2.0); hardness.add(7.0); hardness.add(6.0); hardness.add(10.0);
//        Pair<Double , Double> pair = getTrendDataSimple(hardness ) ;
//        double slope = pair.getKey();
//        double Rss = pair.getValue() ;
//        System.out.println("slope: " + slope);
//        System.out.println("standard Error: " + Rss);
//
//        System.out.println("Jowil Param: "+calcJowilParam(slope, Rss)) ;
//


//        System.out.println(URLDecoder.decode(Test.class.getResource("/reports").getPath(),"utf-8"));



        //        ////////////////////////////  conversion code ///////////////////////////////////////
//        String html = "<html><head><title>Import me</title></head><body><p>Hello World!</p></body></html>";
//
//        File file = new File("E:\\work\\Jowil\\Jowil-Stats\\src\\main\\resources\\htmlReports\\report2.html");
//        Document doc = Jsoup.parse(file, "UTF-8");
//
//        String html = doc.outerHtml();
//
//        WordprocessingMLPackage wordMLPackage = WordprocessingMLPackage.createPackage();
//        AlternativeFormatInputPart afiPart = new AlternativeFormatInputPart(new PartName("/hw.html"));
//        afiPart.setBinaryData(html.getBytes());
//        afiPart.setContentType(new ContentType("text/html"));
//        Relationship altChunkRel = wordMLPackage.getMainDocumentPart().addTargetPart(afiPart);
//
//// .. the bit in document body
//        CTAltChunk ac = Context.getWmlObjectFactory().createCTAltChunk();
//        ac.setId(altChunkRel.getId() );
//        wordMLPackage.getMainDocumentPart().addObject(ac);
//
//// .. content type
//        wordMLPackage.getContentTypeManager().addDefaultContentType("html", "text/html");
//        wordMLPackage.save(new java.io.File("E:\\work\\Jowil\\temp\\test.docx"));

//        final String reportsPath=  "E:\\work\\Jowil\\Jowil-Stats\\src\\main\\resources\\reports\\";
//
//
//        final String report2TemplatePath = reportsPath + "report5\\test.html";
//
//        File file = new File(report2TemplatePath);
//        Document doc = Jsoup.parse(file, "UTF-8");
//
////        doc.select("div#footer").remove();
//        doc.select("p.group-name").last().select("span").last().text("hi");
//        String templateBodyHtml = doc.select("p.group-name").last().select("span").last().text();
//
//        System.out.println(templateBodyHtml);


//        String filePath = ".\\src\\test\\TestCSVs\\welloAnswerKeys.csv" ;
//        BufferedReader input = new BufferedReader(new FileReader(filePath));
//        String line ;
//        ArrayList<ArrayList<String>> csvRows = new ArrayList<>();
//
//        while( (line = input.readLine()) != null ) {
//            String[] row = line.split(",") ;
//            ArrayList<String> rowList = new ArrayList<String>() ;
//            for(int i = 0 ; i < row.length ; i++)
//                rowList.add(row[i]);
//            csvRows.add(rowList);
//        }
//        System.out.println(csvRows);


//        DecimalFormat format = new DecimalFormat("0.##");
//
//        System.out.println(format.format(0));

        /////////////////// test point biserial ////////////////////////////////////
//        double[] a = {1.0 , 1.0 , 0 , 0,  0} ;
//        double [] b = {1.0 , 1.0 , 0 , 1.0,  0} ;
//        double [] c = {1.0 , 1.0 , 1.0 , 0,  1.0 } ;
//        double [] x = {0 , 0 , 1.0 , 1.0 , 1.0 ,} ;
//        double[] total = {3.0 , 3.0 , 1.0, 1.0 , 1.0} ;
//
//        double [] zeros = {1,1,1,1,0};
//
//        ArrayList<Double> hi = new ArrayList<>();
//        hi.add(1.0);hi.add(1.0);hi.add(1.0);hi.add(1.0);hi.add(1.0);
//
//        PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation() ;
//        System.out.println(Double.isNaN(pearsonsCorrelation.correlation( total , zeros))?0:pearsonsCorrelation.correlation( total , zeros)) ;



        //////////////////////// to create pdf from html //////////////////////////////////

//         final String reportsPath=  "E:\\work\\Jowil\\Jowil-Stats\\src\\main\\resources\\reports\\";
//        ReportsHandler reportsHandler = new ReportsHandler();
////        Report1 r1 = new Report1() ;
//        reportsHandler.generatePDF(reportsPath + "report1\\ Report1 - Grades Distribution Report.html", reportsPath + "report1\\test.pdf");




    }
}

