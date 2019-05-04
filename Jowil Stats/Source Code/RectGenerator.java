package Jowil;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.*;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import org.w3c.dom.css.Rect;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class RectGenerator  extends Application {

    final static String COLOR_FOLDER_NAME="gray/";

    @Override
    public void start(Stage primaryStage) throws Exception {

        generateRects(primaryStage,"#095c90","#64686b");
//        generateRoundRects(primaryStage,"#08436b","#095c90");
        Platform.exit();
    }

    void generateRects(Stage stage,String borderColor,String fillColor) throws IOException {

        HBox wrapper=new HBox();
        Rectangle filled=new Rectangle();
        Rectangle empty=new Rectangle();
        wrapper.getChildren().addAll(filled,empty);

        wrapper.setStyle("-fx-border-color:"+borderColor+";-fx-border-width:1");
        filled.setStyle("-fx-fill:"+fillColor+";-fx-border-width:0");
        empty.setStyle("-fx-fill:transparent;-fx-border-width:0");

        filled.setHeight(15);
        empty.setHeight(filled.getHeight());
        stage.setScene(new Scene(wrapper));


        for(int i=0;i<=100;i++){
            filled.setWidth(i);
            empty.setWidth(100-i);
            WritableImage snapShot = wrapper.snapshot(new SnapshotParameters() , null);
            ImageIO.write(SwingFXUtils.fromFXImage(snapShot, null), "png",
                    new File("./src/main/resources/RectImages/Report5/"+COLOR_FOLDER_NAME+Integer.toString(i)+".png"));
        }

    }

    void generateRoundRects(Stage stage,String borderColor,String fillColor) throws IOException {
        StackPane stack=new StackPane();
        Rectangle wrapper=new Rectangle();
        Rectangle filled=new Rectangle();
        Rectangle empty=new Rectangle();


        wrapper.setStyle("-fx-stroke:"+borderColor+";-fx-stroke-width:2;-fx-fill:white;-fx-stroke-line-cap:round;-fx-stroke-line-join:round;-fx-stroke-type:outside");
        HBox innerHbox=new HBox();

        innerHbox.setStyle("-fx-background-color:transparent");
        double padValue=2;
        innerHbox.setPadding(new Insets(0,padValue,0,padValue));
        innerHbox.getChildren().addAll(filled,empty);
        stack.getChildren().addAll(wrapper,innerHbox);
        filled.setStyle("-fx-fill:"+fillColor+";-fx-stroke-width:0");
        empty.setStyle("-fx-fill:white;-fx-stroke-width:0");
        int widthFactor=3;
        wrapper.setWidth(100*widthFactor);
        wrapper.setHeight(12);
        wrapper.setArcWidth(12);
        wrapper.setArcHeight(12);
        double shiftValue=2;
        StackPane.setMargin(innerHbox,new Insets(shiftValue,0,0,0));
        filled.setArcWidth(wrapper.getArcWidth());
        filled.setArcHeight(wrapper.getArcHeight());
        empty.setArcWidth(wrapper.getArcWidth());
        empty.setArcHeight(wrapper.getArcHeight());
        filled.setHeight(wrapper.getHeight());
        empty.setHeight(filled.getHeight());
        stage.setScene(new Scene(stack));


        for(int i=0;i<=100;i++){
            filled.setWidth(i*widthFactor);
            empty.setWidth(wrapper.getWidth()-filled.getWidth());
            WritableImage snapShot = stack.snapshot(new SnapshotParameters() , null);
            ImageIO.write(SwingFXUtils.fromFXImage(snapShot, null), "png",
                    new File("./src/main/resources/RectImages/Report4/"+Integer.toString(i)+".png"));
        }



    }
}