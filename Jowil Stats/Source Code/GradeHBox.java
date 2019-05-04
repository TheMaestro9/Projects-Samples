package Jowil;

import com.jfoenix.controls.JFXSlider;
import com.jfoenix.controls.JFXTextField;
import de.jensd.fx.glyphs.GlyphsBuilder;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIcon;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIconView;
import javafx.beans.property.DoubleProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Alert;
import javafx.scene.control.Label;
import javafx.scene.control.Tooltip;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.text.Font;
import javafx.util.Pair;

import java.util.ArrayList;

import static Jowil.Controller.constructMessage;
import static Jowil.Controller.resX;


public class GradeHBox extends HBox {

    private int index;

    JFXTextField nameTextField=new JFXTextField();
    JFXTextField percentScoreTextField=new JFXTextField();
    Label rawScoreTextField=new Label();
    JFXSlider scoreSlider=new JFXSlider();
    private final  Node addIcon=GlyphsBuilder.create(FontAwesomeIconView.class).glyph(FontAwesomeIcon.PLUS).size(Double.toString(resX/85)).styleClass("gradesPlusIcon").build();
    private final  Node removeIcon=GlyphsBuilder.create(FontAwesomeIconView.class).glyph(FontAwesomeIcon.TRASH_ALT).size(Double.toString(resX/80)).styleClass("projectRemoveIcon").build();
    StackPane addButton=new StackPane();
    StackPane removeButton=new StackPane();
    boolean isIgnorePercentScoreLosingFocus=false;

    GradeBoundariesController parentController;




    GradeHBox(int index,String name,String percentScore,GradeBoundariesController parentController){

        this.index=index;
        this.parentController=parentController;



        this.nameTextField.setText(name);
        nameTextField.setStyle("-jfx-focus-color:#095c90;-fx-font-size:"+Double.toString(resX/100));
        nameTextField.setAlignment(Pos.CENTER);
        this.nameTextField.textProperty().addListener(t-> {
            parentController.setContentEdited(true);
            parentController.isNewScaleSavedBefore=false;
        });

        initPercentScoreTextField(percentScore);
        initRawScoreTextField(percentScore);


        this.scoreSlider.setValue(Double.parseDouble(percentScore));
        scoreSlider.valueProperty().addListener((observable,oldValue,newValue)-> {

            double d=(Double)newValue;
            this.percentScoreTextField.setText(String.format("%.1f",d));
            this.rawScoreTextField.setText(String.format("%.1f",d/ 100 * Statistics.getMaxScore()));

        });

        scoreSlider.setBlockIncrement(1.0);




        addButton.getChildren().add(addIcon);
        addButton.setOnMouseClicked(t-> parentController.addNextGrade(this.index));
        TranslatableTooltip tooltipAdd = new TranslatableTooltip("Add Grade Below");
        Tooltip.install(addButton, tooltipAdd);

        removeButton.getChildren().add(removeIcon);
        removeButton.setOnMouseClicked(t-> parentController.deleteGrade(this.index));
        TranslatableTooltip tooltipRemove = new TranslatableTooltip("Delete Grade");
        Tooltip.install(removeButton, tooltipRemove);


        this.getChildren().addAll(nameTextField,percentScoreTextField,rawScoreTextField,scoreSlider,addButton,removeButton);


    }


    GradeHBox(GradeHBox orig){

        this(orig.index,orig.nameTextField.getText(),orig.percentScoreTextField.getText(),orig.parentController);
    }

    public void updateSizes(double scrollPaneWidth,double scrollPaneHeight){


        nameTextField.setPrefWidth((int)(scrollPaneWidth*0.15));
        nameTextField.setPrefHeight((int)(scrollPaneHeight/20));
        rawScoreTextField.setPrefWidth((int)(scrollPaneWidth*0.13));
        rawScoreTextField.setPrefHeight((int)(scrollPaneHeight/20));
        percentScoreTextField.setPrefWidth((int)(scrollPaneWidth*0.13));
        percentScoreTextField.setPrefHeight((int)(scrollPaneHeight/20));
        scoreSlider.setPrefWidth((int)(scrollPaneWidth*0.3));
        scoreSlider.setPrefHeight((int)resX*14/680);
        scoreSlider.setPadding(new Insets((int)((rawScoreTextField.getPrefHeight()/2)),0,0,0));
        rawScoreTextField.setPadding(new Insets((int)((rawScoreTextField.getPrefHeight()/3)),0,0,0)); //label option
        rawScoreTextField.setStyle("-fx-font-size:"+resX*14/1280);//label option
        this.setSpacing((int)(scrollPaneWidth*0.03));

    }


    public void incrementIndex() {
        index++;
    }


    public void decrementIndex() {
        index--;
    }

    private void initPercentScoreTextField(String percentScore){

        this.percentScoreTextField.setText(percentScore);
        percentScoreTextField.setStyle("-jfx-focus-color:#095c90;-fx-font-size:"+Double.toString(resX/100));
        percentScoreTextField.setAlignment(Pos.CENTER);


        percentScoreTextField.setOnKeyPressed(event -> {

            if(event.getCode()==KeyCode.ENTER){

                if(isPercentScoreTextFieldInValid()){
                    isIgnorePercentScoreLosingFocus=true;
                    parentController.showAlertAndWait(Alert.AlertType.ERROR, parentController.stage.getOwner(), "Grade Scale Value Error",
                            constructMessage("Grade scale value"," \""+percentScoreTextField.getText(),"\" ","is invalid."));
                    percentScoreTextField.setText(String.format("%.1f",scoreSlider.getValue()));
                }
                else {
                    parentController.rootPane.requestFocus();
                    scoreSlider.setValue(Double.parseDouble(percentScoreTextField.getText()));
                    rawScoreTextField.setText(String.format("%.1f", Double.parseDouble(percentScoreTextField.getText()) / 100 * Statistics.getMaxScore()));
                }

            }
        });

        percentScoreTextField.setOnKeyTyped(event-> {
            if(Character.isDigit(event.getCharacter().charAt(0)) || (event.getCharacter().equals(".") && !percentScoreTextField.getText().contains("."))) //invalid character
                return;

            event.consume();
        });

        percentScoreTextField.focusedProperty().addListener((observable,oldValue,newValue)-> {

            if(isIgnorePercentScoreLosingFocus){
                isIgnorePercentScoreLosingFocus=false;
                return;
            }

            if(newValue)
                return;

            if(isPercentScoreTextFieldInValid()){
                parentController.showAlertAndWait(Alert.AlertType.ERROR, parentController.stage.getOwner(), "Grade Scale Value Error",
                        constructMessage("Grade scale value"," \""+percentScoreTextField.getText(),"\" ","is invalid."));
                percentScoreTextField.setText(String.format("%.1f",scoreSlider.getValue()));
            }
            else {
                scoreSlider.setValue(Double.parseDouble(percentScoreTextField.getText()));
                rawScoreTextField.setText(String.format("%.1f", Double.parseDouble(percentScoreTextField.getText()) / 100 * Statistics.getMaxScore()));
            }

        });

        percentScoreTextField.textProperty().addListener(t-> {
            parentController.setContentEdited(true);
            parentController.isNewScaleSavedBefore=false;
        });

    }



    private void initRawScoreTextField(String percentScore){

        this.rawScoreTextField.setText(String.format("%.1f",Double.parseDouble(percentScore)/100*Statistics.getMaxScore()));
        this.rawScoreTextField.setStyle("-jfx-focus-color:#095c90;-fx-font-size:"+Double.toString(resX/100));
        rawScoreTextField.setAlignment(Pos.CENTER);
        //rawScoreTextField.setEditable(false); label option

        rawScoreTextField.focusedProperty().addListener((observable,oldValue,newValue)->{
            if(newValue)
                parentController.rootPane.requestFocus();
        });


    }


    private boolean isPercentScoreTextFieldInValid(){
        String s=percentScoreTextField.getText();
        return s.length()==0 ||  s.equals(".")||Double.parseDouble(s)>100;
    }

    public Pair<String,Double> getGrade(){
        return new Pair(nameTextField.getText(),Double.parseDouble(percentScoreTextField.getText()));
    }

    public void refreshRawScore(){
        rawScoreTextField.setText(String.format("%.1f", Double.parseDouble(percentScoreTextField.getText()) / 100 * Statistics.getMaxScore()));
    }



}
