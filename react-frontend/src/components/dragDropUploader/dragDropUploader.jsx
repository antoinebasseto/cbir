import { FileUploader } from "react-drag-drop-files";
import "./dragDropUploader.css"
import {FiUpload} from "react-icons/fi"

const fileTypes = ["JPG", "PNG", "GIF"];
const children = 
    <div 
        className="py-4 px-6 my-2 space-x-2 flex items-center w-fit max-w-lg rounded-2xl shadow-xl text-center font-sans font-semibold text-lg bg-zinc-100 hover:bg-zinc-200 active:bg-zinc-300">
        <FiUpload className="h-4/5"/>
        <h5 className="">
            UPLOAD
        </h5>
    </div>



export default function DragDropUploader(props) {
    return (
        <FileUploader 
            handleChange={props.onImageUploadedChange} 
            name="file" 
            types={fileTypes} 
            children={children}
        />
    );
}