package Jowil;


import com.jfoenix.controls.JFXCheckBox;
import de.jensd.fx.glyphs.GlyphsBuilder;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIcon;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIconView;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;

import static Jowil.Controller.isTranslationMode;
import static Jowil.Controller.resX;


public class TreeViewCustomCell extends TreeCell<String> {

    private final double CHECK_BOXES_SIZE=resX*14/1280;

    @Override
    protected void updateItem(String item, boolean empty) {
        super.updateItem(item, empty);

        this.selectedProperty().addListener((observable,oldValue,newValue)->{if(newValue) this.updateSelected(false);});
        this.focusedProperty().addListener((observable,oldValue,newValue)->this.setFocused(false));



        if (isEmpty()) {
            setGraphic(null);
            setText(null);
        } else {

            if (this.getTreeItem().isLeaf()) {


                //HBox cellBox = new HBox(0);
                boolean isBinaryGroup=this.getTreeItem().getParent().getChildren().size()==2 &&
                        (item.toLowerCase().equals("t") || item.toLowerCase().equals("f"));

                String checkBoxText="";

                if(Controller.isTranslateFormContent)
                    checkBoxText=isBinaryGroup?Translator.englishToArabicBinary(item):Translator.englishToArabic(item);
                else
                    checkBoxText=item;

                JFXCheckBox checkBox = new JFXCheckBox(checkBoxText,CHECK_BOXES_SIZE);
                this.setStyle("-fx-font-size:"+resX*12/1280);
//                Label label = new Label(item);
//                label.setPadding(new Insets(0,0,0,0));
                checkBox.setPadding(new Insets(0,0,0,0));
                checkBox.setSelected(GroupsController.isChoicePossible(this.getTreeItem().getParent().getParent().getValue(), this.getTreeItem().getValue()));

                checkBox.selectedProperty().addListener((observable,oldValue,newValue)->{

                    String groupName=this.getTreeItem().getParent().getParent().getValue();
                    if(newValue) {
                        GroupsController.restoreToGroup(groupName, this.getTreeItem().getValue());
                        this.getTreeView().refresh();
                        this.getTreeView().requestFocus();
                    }
                    else if(GroupsController.getLastPossible(groupName)==GroupsController.getFirstPossible(groupName))
                        checkBox.setSelected(true);
                    else{
                        GroupsController.deleteFromGroup(groupName, this.getTreeItem().getValue());
                        this.getTreeView().refresh();
                        this.getTreeView().requestFocus();
                    }

                });


//                label.prefHeightProperty().bind(checkBox.heightProperty());
                //cellBox.getChildren().addAll(checkBox,label);

                setGraphic(checkBox);
                setText(null);

            } else if(isGroupNameNode()) {


                HBox cellBox = new HBox(4);
                cellBox.setAlignment(Pos.CENTER_LEFT);
                Label label = new Label(item);
                this.setStyle("-fx-font-size:"+resX*15/1280+";-fx-font-weight: bold;-fx-text-fill:#5e5c5c");
                Node plusIcon=GlyphsBuilder.create(FontAwesomeIconView.class).glyph(FontAwesomeIcon.PLUS_CIRCLE).size(Double.toString(resX/95)).styleClass("groupPlusIcon").build();
                StackPane iconPane=new StackPane();
                iconPane.getChildren().add(plusIcon);

                this.setId(this.getTreeItem().getValue());
                iconPane.setOnMouseReleased(t->{
                    GroupsController.addToGroup(this.getTreeItem().getValue());
                    System.out.println(this.getTreeItem().getValue());
                    this.getTreeView().refresh();
                });


                //Node d=this.getTreeItem().getGraphic().lookup("habeby");


                cellBox.getChildren().addAll(label,iconPane);

                setGraphic(cellBox);
                setText(null);
            }else{ //range node

                HBox cellBox = new HBox(resX/400);
                cellBox.setAlignment(Pos.CENTER_LEFT);
                String min=this.getTreeItem().getChildren().get(GroupsController.getFirstPossible(this.getTreeItem().getParent().getValue())).getValue();
                String max=this.getTreeItem().getChildren().get(GroupsController.getLastPossible(this.getTreeItem().getParent().getValue())).getValue();

                boolean isBinaryGroup=this.getTreeItem().getChildren().size()==2;
                String minLabelText="";
                String maxLabelText="";

                if(Controller.isTranslateFormContent) {
                    minLabelText = isBinaryGroup ? Translator.englishToArabicBinary(min) : Translator.englishToArabic(min);
                    maxLabelText = isBinaryGroup ? Translator.englishToArabicBinary(max) : Translator.englishToArabic(max);
                }
                else{
                    minLabelText=min;
                    maxLabelText=max;
                }

                Label minLabel=new Label(minLabelText);
                Label maxLabel=new Label(maxLabelText);
                minLabel.setPadding(new Insets(0,0,0,0));
                maxLabel.setPadding(new Insets(0,0,0,0));
                Node rightArrowIcon=GlyphsBuilder.create(FontAwesomeIconView.class).glyph(isTranslationMode?FontAwesomeIcon.ARROW_LEFT:FontAwesomeIcon.ARROW_RIGHT).size(Double.toString(resX/111)).styleClass("error").build();
                this.selectedProperty().addListener((observable,oldValue,newValue)->{
                    if(newValue)
                        rightArrowIcon.setStyle("-fx-fill:white");
                    else
                        rightArrowIcon.setStyle("-fx-fill:#095c90");
                });
                rightArrowIcon.setStyle("-fx-fill:#095c90");

                this.setStyle("-fx-font-size:"+resX*14/1280);
                cellBox.getChildren().addAll(minLabel,rightArrowIcon,maxLabel);
                setGraphic(cellBox);
                setText(null);
            }
        }
    }

    protected boolean isGroupNameNode(){
        return this.getTreeItem().getParent()==this.getTreeView().getRoot();
    }
}