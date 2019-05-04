package Jowil;

import Jowil.Reports.*;
import com.jfoenix.controls.JFXSpinner;
import com.lowagie.text.DocumentException;
import javafx.application.Platform;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.NodeOrientation;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.pdfsam.ui.FillProgressIndicator;
import org.pdfsam.ui.RingProgressIndicator;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Optional;

import static Jowil.Controller.*;

public class ReportProgressWindow {

    private final ArrayList<Boolean> isGenerateReports;
    Scene scene;
    Stage stage;

    volatile static SimpleIntegerProperty progressCount;
    int reportsCount;
    double resX=Controller.resX;
    double resY=Controller.resY;
    Thread th;
    double rootWidth=1280/2;
    double rootHeight=680/2;

    private static volatile SimpleDoubleProperty reportProgress;

    Pane root;

    ArrayList<Integer> formatsOut;




    RingProgressIndicator counterIndicator;

    @FXML
    FillProgressIndicator reportProgressIndicator;


    @FXML
    VBox counterVBox;

    @FXML
    HBox mainHBox;

    @FXML
    Label titleLabel;


    static JFXSpinner spinner;

    ReportsHandler reportsHandler;


    ReportProgressWindow(int reportsCount,ArrayList<Boolean> isGenerateReports,ArrayList<Integer>formatsOut){

        spinner=new JFXSpinner();

        reportProgress=new SimpleDoubleProperty();
        progressCount=new SimpleIntegerProperty();

        reportsHandler = new ReportsHandler();


        this.reportsCount=reportsCount;
        this.formatsOut=formatsOut;
        this.isGenerateReports=isGenerateReports;



        reportProgress.addListener(t -> {

            Platform.runLater(() -> {
                reportProgressIndicator.setProgress((reportProgress.get()*100));

                //counterIndicator.setProgress((int)((reportProgress.get()+progressCount.get())/(double)reportsCount*100));
            });
        });

        progressCount.addListener(t -> {

            double progressValue = ((double) progressCount.get() / reportsCount *100);

            Platform.runLater(() -> {
                counterIndicator.setProgress(progressValue);
            });

            if(progressValue==100) { //all reports finished -> animate finish and open explorer
                try {
                    Thread.sleep(1200);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                Platform.runLater(() -> showFolderInExplorer());
            }

        });

    }



    public void initialize() {



        mainHBox.setLayoutX((int)(rootWidth*0.18));
        mainHBox.setLayoutY((int)(rootHeight*0.25));
        mainHBox.setSpacing((int)(1280/20));

        spinner.setLayoutX(rootWidth*0.9);
        spinner.setLayoutY(rootHeight*0.075);
        spinner.setPrefSize(30,30);
        spinner.setVisible(true);
        spinner.getStyleClass().add("blueSpinner");

        counterIndicator=new RingProgressIndicator(reportsCount);

        counterIndicator.setNodeOrientation(NodeOrientation.LEFT_TO_RIGHT); //override translation behaviour
        counterVBox.getChildren().add(counterIndicator);

        counterIndicator.setStyle("-fx-background-color:transparent");

        titleLabel.setLayoutY((int)(680/18));
        titleLabel.setLayoutX((int)(1280/70));

        stage.setOnCloseRequest(event->{
            boolean isQuit=Controller.showConfirmationDialog("Quit Reports Generation","Are you sure you want to quit reports generation?",stage.getOwner());
            if(isQuit) {//interrupt is used for stopping when pdf is included and the boolean is used when no boolean is used
                th.interrupt();
                reportsHandler.setIsStopReportsGeneration(true);
            }
            else
                event.consume();
        });



    }

    public void startWindow(){
        try {


            FXMLLoader loader = new FXMLLoader(getClass().getResource("/FXML/reportProgress.fxml"));
            stage = new Stage();
            loader.setController(this);
            root= loader.load();
            if(isTranslationMode)
                root.setNodeOrientation(NodeOrientation.RIGHT_TO_LEFT);

            //root.getChildren().add(spinner);
            scene = new Scene(root,rootWidth,rootHeight);
            String windowTitle="Reports Generation Progress";
            stage.setTitle(Controller.isTranslationMode &&translations.containsKey(windowTitle)?translations.get(windowTitle):windowTitle);
            stage.setScene(scene);
            stage.setResizable(false);
            stage.initModality(Modality.APPLICATION_MODAL);
            stage.show();
            Controller.translateAllNodes(scene.getRoot());

        }
        catch (IOException e) {
            e.printStackTrace();
        }



        generateReports();

    }



    public static void incrementProgressCount(){
        progressCount.setValue(progressCount.get()+1);

    }


    public static void setReportProgress(double reportProgress) {
        ReportProgressWindow.reportProgress.set(reportProgress);
    }

    public static void removeSpinner(){
        spinner.setVisible(false);
    }


    private void generateReports(){

        Runnable task =()->{

            ArrayList<Report> reportsOut=new ArrayList<>();

            if(isGenerateReports==null) //questMode
                reportsOut.add(new Report9());
            else {
                for (int i = 0; i < isGenerateReports.size(); i++) {
                    if (isGenerateReports.get(i))
                        reportsOut.add(getReport(i + 1));
                }
            }



            reportsHandler.setIsStopReportsGeneration(false);

            try {
                reportsHandler.generateReports(reportsOut, formatsOut);
            } catch (IOException e) {
                int cropIndex=e.getMessage().indexOf("(The");
                String fileName=e.getMessage().substring(0,cropIndex==-1?1:cropIndex);
                String extraStat=constructMessage(" Make sure that the file"," \""+fileName+"\" ", "is not opened in another application.");
                showReportsErrorMessage(cropIndex==-1?e.getMessage():extraStat);

            } catch (DocumentException e) {
                showReportsErrorMessage("");
                e.printStackTrace();
            }
            catch(RuntimeException e) {
                if(!Thread.currentThread().isInterrupted()) //not caused by interruption
                    showReportsErrorMessage("");
                e.printStackTrace();
            } catch (InvalidFormatException e) {
                e.printStackTrace();
            }

        };



        th = new Thread(task);
        th.setDaemon(false);
        th.start();

    }

    private void deleteLastReport(){
//        String fileName = Report.getPDFPath() + "Report" + (progressCount.get() + 1) + ".pdf";
//        System.out.println(fileName);
//        System.out.println("Delete: " + new File(fileName).delete());
    }

    private void showReportsErrorMessage(String extraStatement){

        Platform.runLater(()->{

        Alert alert = new Alert(Alert.AlertType.ERROR);


        alert.getButtonTypes().add(ButtonType.CLOSE);
        Button closeButt=(Button)alert.getDialogPane().lookupButton(ButtonType.CLOSE);
        closeButt.setText("Close");

        Button yessButt=(Button)alert.getDialogPane().lookupButton(ButtonType.OK);
        yessButt.setText("Retry");

        alert.setTitle("Reports Generation Error");
        alert.setHeaderText(null);
        alert.setContentText(constructMessage("An error has occurred during report generation.",extraStatement));
        alert.initOwner(stage.getOwner());
        processDialog(alert);
        Optional<ButtonType> option = alert.showAndWait();
        stage.hide();

        if(option.get()==ButtonType.OK)
            new ReportProgressWindow(reportsCount,isGenerateReports,formatsOut).startWindow();

        stage.close();
        });

    }

    private void showFolderInExplorer(){

        stage.hide();

        File file = new File (Report.getOutPath());
        Desktop desktop = Desktop.getDesktop();

        int i=0; //try to open 5 times at most -> if failed display error message


            try {
                desktop.open(file);

            } catch (IOException  | IllegalArgumentException e) {
                Controller.showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Directory Error", "Cannot open reports directory.");

            }


        stage.close();

    }


    private Report getReport(int reportNumber){

        switch (reportNumber){
            case 1:
                return new Report1();
            case 2:
                return new Report2();
            case 3:
                return new Report3();
            case 4:
                return new Report4();
            case 5:
                return new Report5();
            case 6:
                return new Report6();
            case 7:
                return new Report7();
            case 8:
                return new Report8();
        }
        return null;
    }



}
