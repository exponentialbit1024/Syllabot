import { Component } from 'react';

class Callback extends Component {

  constructor() {
    super()
  }

  componentDidMount() {
    console.log(this.props);
  }

  render() {
    return null;
  }
}

export default Callback;
