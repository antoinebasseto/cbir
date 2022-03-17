import React from 'react';


class ImageChoiceComponent extends React.Component<{}, {imagePath: string}> {

  constructor(props: any) {
    super(props);
    this.state = {imagePath: ''};
    this.handleClick = this.handleClick.bind(this);
  }

  render() {
    return (
      <div>
        <button onClick={this.handleClick}>
          Display image 
        </button>
        <img src={this.state.imagePath} />
      </div>
    );
  }

  handleClick(e : any){
    this.setState({
      imagePath: "./logo.svg",
    });            
  }
}
export default ImageChoiceComponent;
